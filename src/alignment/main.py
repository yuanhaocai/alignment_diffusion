import os
import torch
from torch.utils.data import DataLoader
import argparse
from tqdm import tqdm
try:
    from accelerate import Accelerator
except ImportError:
    class Accelerator:
        def __init__(self):
            self.is_main_process = True
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        def prepare(self, *args):
            prepared = []
            for arg in args:
                if isinstance(arg, torch.nn.Module):
                    prepared.append(arg.to(self.device))
                else:
                    prepared.append(arg)
            return tuple(prepared) if len(prepared) > 1 else prepared[0]

        def backward(self, loss):
            loss.backward()

        def gather_for_metrics(self, value):
            return value.detach().reshape(-1).cpu()
import json

from evaluate import *
from save import *
from model import *
from data import *


def main(args):
    """
    The main training and evaluation loop.
    """
    accelerator = Accelerator()

    print("Arguments:")
    for arg, value in vars(args).items():
        print(f"  {arg}: {value}")

    # Exit if training has already been done
    loss_files = ['train_loss.json', 'train_loss.csv', 'train_loss_log.csv']
    loss_paths = [os.path.join(args.save_dir, f) for f in loss_files]

    if any(os.path.exists(path) for path in loss_paths):
        if accelerator.is_main_process:
            print(f"Detected saved training loss. Training already completed. Exiting.")
        return  # or sys.exit(0)

    if accelerator.is_main_process:
        save_hyperparams(args, args.save_dir)

    train_dataset = Dataset_text_tabular(
        args.text_path, 
        args.train_z_dat_path, 
        args.y_path, 
        args.y_type, 
        'train'
    )
    # Stage 1 model selection is validation-only. The test split is never
    # loaded by the alignment trainer.
    eval_dataset = Dataset_text_tabular(
        args.text_path,
        args.train_z_dat_path,
        args.y_path,
        args.y_type,
        'val',
    )
    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        drop_last=True,
        num_workers=args.num_workers,
        pin_memory=True,
        persistent_workers=(args.num_workers > 0)
    )
    eval_loader = DataLoader(
        eval_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        drop_last=False,
        num_workers=args.num_workers,
        pin_memory=True,
        persistent_workers=(args.num_workers > 0)
    )
    model = TabTextAlign(
        emb_dim=args.emb_dim, 
        num_classes=args.y_num_classes,
        y_type=args.y_type,
        projector_mode=args.projector_mode,
        residual_scale=args.residual_scale,
    )
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    model, optimizer, train_loader = accelerator.prepare(model, optimizer, train_loader)
    
    train_loss_log = []
    val_loss_log = []
    best_val_loss = float("inf")
    patience = args.patience
    patience_counter = 0
    best_epoch = -1
    y_class_weight = torch.tensor(args.y_class_weight, device=accelerator.device) if args.y_class_weight is not None else None

    out_weights_dir = os.path.join(args.save_dir, "models")
    eval_dir = os.path.join(args.save_dir, "eval")
    is_regression = args.y_type == 'continuous'

    for epoch in range(args.n_epochs):
        model.train()
        train_losses = []
        dat_text_losses = []
        sup_losses = []
        for batch in tqdm(train_loader, disable=not accelerator.is_main_process):
            tab_dat_emb, y, text_emb = batch
            tab_dat_emb = tab_dat_emb.float().to(accelerator.device)
            text_emb = text_emb.float().to(accelerator.device)
            y = y.to(accelerator.device)
            tab_dat_proj, text_proj, logit_scale = model(tab_dat_emb, text_emb)

            sim_dat_text = compute_combined_cosine_sim(tab_dat_proj, text_proj)
            loss_dat_text = clip_loss(sim_dat_text, logit_scale)

            # logits = unwrap_model(model).classifier(tab_dat_proj)  # [B,5]
            pred = unwrap_model(model).predict_response(tab_dat_proj, text_proj)
            if is_regression:
                pred = pred.squeeze(-1)  # [B]
                sup_loss = torch.nn.functional.mse_loss(pred, y)
                # maybe Clip loss weight: for regression, you could set it lower if it's a regression-only scenario.
            else:
                sup_loss = torch.nn.functional.cross_entropy(pred, y, weight=y_class_weight)

            loss = (args.weight_clip_loss * loss_dat_text
                    + args.weight_sup_loss * sup_loss)

            optimizer.zero_grad()
            accelerator.backward(loss)
            optimizer.step()

            # Gather for metrics and accumulate (detach first)
            train_losses.append(accelerator.gather_for_metrics(loss.detach()))
            dat_text_losses.append(accelerator.gather_for_metrics(loss_dat_text.detach()))
            sup_losses.append(accelerator.gather_for_metrics(sup_loss.detach()))

        # --- Averaging train loss on all processes ---
        train_loss = torch.cat(train_losses).float().mean().item()
        dt_loss = torch.cat(dat_text_losses).float().mean().item()
        sup_loss = torch.cat(sup_losses).float().mean().item()
        loss_dict = {
            'train_loss': float(train_loss),
            'clip_loss': float(dt_loss),
            'sup_loss': float(sup_loss),
        }
        train_loss_log.append(loss_dict)
        
        # === Validation/Evaluation step ===
        if epoch % args.eval_interval == 0 or epoch == args.n_epochs - 1:
            model.eval()
            val_losses = []
            if accelerator.is_main_process:
                metrics = evaluate(model, eval_loader, args.y_type, args.y_num_classes)

            with torch.no_grad():
                for batch in eval_loader:
                    tab_dat_emb, y, text_emb = batch
                    tab_dat_emb = tab_dat_emb.float().to(accelerator.device)
                    text_emb = text_emb.float().to(accelerator.device)
                    y = y.to(accelerator.device)
                    tab_dat_proj, text_proj, logit_scale = model(tab_dat_emb, text_emb)

                    sim_dat_text = compute_combined_cosine_sim(tab_dat_proj, text_proj)
                    loss_dat_text = clip_loss(sim_dat_text, logit_scale)

                    # logits = unwrap_model(model).classifier(tab_dat_proj)
                    pred = unwrap_model(model).predict_response(tab_dat_proj, text_proj)
                    if is_regression:
                        pred = pred.squeeze(-1)  # [B]
                        sup_loss = torch.nn.functional.mse_loss(pred, y)
                    else:
                        sup_loss = torch.nn.functional.cross_entropy(pred, y, weight=y_class_weight)
                    val_loss = (args.weight_clip_loss * loss_dat_text
                                + args.weight_sup_loss * sup_loss)

                    # Gather for metrics and accumulate
                    val_losses.append(accelerator.gather_for_metrics(val_loss.detach()))

            val_losses = torch.cat(val_losses)
            val_loss = val_losses.float().mean().item()
            val_loss_log.append(float(val_loss))
            
            if accelerator.is_main_process:
                # Save metrics and append to CSV for each eval epoch
                save_metrics({**metrics, "train_loss": train_loss, "val_loss": val_loss, "epoch": epoch}, eval_dir, epoch)
                append_metrics_to_csv({**metrics, "train_loss": train_loss, "val_loss": val_loss, "epoch": epoch}, eval_dir)
                save_model(model, out_weights_dir, epoch)

                # Save best model so far
                # Pick validation loss (or negative macroF1) as main criterion
                if val_loss < best_val_loss:
                    best_val_loss = val_loss
                    best_epoch = epoch
                    patience_counter = 0
                    save_model(model, out_weights_dir)
                    print(f"New best model saved at epoch {epoch}, val_loss={val_loss:.4f}")
                else:
                    patience_counter += 1
                    print(f"No improvement. {patience_counter}/{patience} epochs since best val loss.")
                # Early stopping
                if patience_counter >= patience:
                    print(f"Early stopping at epoch {epoch} due to no improvement in validation loss for {patience} epochs.")
                    break
                print(f"[Eval][Epoch {epoch}] ", end="")
                print(", ".join([f"{k}={metrics[k]:.4f}" for k in metrics if isinstance(metrics[k], (int, float))]))

    # Final model and log saving
    if accelerator.is_main_process:
        best_model_path = os.path.join(out_weights_dir, "tabtext_align.pt")
        best_state = torch.load(best_model_path, map_location=accelerator.device)
        best_state = {
            key.removeprefix("module."): value for key, value in best_state.items()
        }
        unwrap_model(model).load_state_dict(best_state)
        metrics = evaluate(model, eval_loader, args.y_type, args.y_num_classes)
        metrics['best_epoch'] = best_epoch
        save_metrics(metrics, os.path.join(args.save_dir, "eval"), "final")

        train_loss_path = os.path.join(args.save_dir, 'train_loss.csv')
        train_loss_log = pd.DataFrame(
            train_loss_log,
            columns=['train_loss', 'clip_loss', 'sup_loss']
        )
        train_loss_log.to_csv(train_loss_path, index=False)

        val_loss_path = os.path.join(args.save_dir, 'val_loss.json')
        with open(val_loss_path, 'w') as f:
            json.dump(val_loss_log, f, indent=2)

        print("The best epoch is:", best_epoch)
        print(f"Saved train loss log to {train_loss_path}")
        print(f"Saved val loss log to {val_loss_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--text_path", type=str, required=True)
    parser.add_argument("--train_z_dat_path", type=str, required=True)
    parser.add_argument("--y_path", type=str, required=True)
    parser.add_argument("--save_dir", type=str, required=True)
    parser.add_argument("--batch_size", type=int, default=128)
    parser.add_argument("--emb_dim", type=int, default=768)
    parser.add_argument("--y_num_classes", type=int, default=None)
    parser.add_argument("--y_class_weight", nargs=2, type=float, default=None)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--n_epochs", type=int, default=50)
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--eval_interval", type=int, default=5)
    parser.add_argument("--weight_clip_loss", type=float, default=1.0)
    parser.add_argument("--weight_sup_loss", type=float, default=1.0)
    parser.add_argument("--projector_mode", type=str, default="linear", choices=["linear", "identity_init", "residual", "identity"])
    parser.add_argument("--residual_scale", type=float, default=1.0)
    parser.add_argument("--patience", type=int, default=150)
    parser.add_argument("--y_type", type=str, default=None, required=True)
    args = parser.parse_args()
    main(args)
