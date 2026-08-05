import argparse
from tabsynfnn.tabsyn.main_pm import diffusion_train_pm
from tabsynfnn.prediction_interval_utils import str2bool

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--y_emb_path', required=True, type=str)
    parser.add_argument('--emb_alignment_path', type=str, default=None)
    parser.add_argument('--diffusion_path', required=True, type=str)
    parser.add_argument('--y_type', type=str, required=True)
    parser.add_argument('--use_tab', type=str2bool, nargs='?', const=True, default=True)
    parser.add_argument('--use_text', type=str2bool, nargs='?', const=True, default=True)
    parser.add_argument('--use_image', type=str2bool, nargs='?', const=True, default=True)
    parser.add_argument('--vae_dat_path', type=str, default=None)
    parser.add_argument('--text_path', type=str, default=None)
    parser.add_argument('--image_path', type=str, default=None)
    parser.add_argument('--epochs', type=int, default=5000)
    parser.add_argument('--batch_size', type=int, default=4096)
    parser.add_argument('--lr', type=float, default=1e-4)
    parser.add_argument('--d_in', type=int, default=5)
    parser.add_argument('--dim_t', type=int, default=1024)
    parser.add_argument('--cond_dim', type=int, default=None)
    parser.add_argument('--text_pooling', type=str, choices=['mean', 'first'], default='mean')
    parser.add_argument('--normalize_tab_cond', type=str2bool, nargs='?', const=True, default=True)
    parser.add_argument('--normalize_text_tokens', type=str2bool, nargs='?', const=True, default=False)
    parser.add_argument('--normalize_text_pooled', type=str2bool, nargs='?', const=True, default=True)
    parser.add_argument('--normalize_projected_cond', type=str2bool, nargs='?', const=True, default=False)
    parser.add_argument('--alignment_mix_alpha', type=float, default=1.0)
    parser.add_argument('--raw_vae_dat_path', type=str, default=None)
    parser.add_argument('--raw_text_path', type=str, default=None)
    parser.add_argument('--early_stopping_thrshd', type=int, default=500)
    parser.add_argument('--clip_grad', type=str2bool, nargs='?', const=True, default=True)
    parser.add_argument('--clip_grad_max_norm', type=float, default=2.0)
    parser.add_argument('--diffusion_class_weights', nargs='+', type=float, default=None)
    parser.add_argument('--normalize_diffusion_class_weights', type=str2bool, nargs='?', const=True, default=True)
    parser.add_argument('--save_milestone_epochs', type=str, default=None)
    parser.add_argument('--use_wandb', type=str2bool, nargs='?', const=True, default=True)
    parser.add_argument('--wandb_project', type=str)
    parser.add_argument('--run_group', type=str, default=None)
    parser.add_argument('--run_name', type=str, default=None)
    parser.add_argument('--wandb_log_interval', type=int, default=10)
    parser.add_argument('--resume', type=str2bool, nargs='?', const=True, default=False)
    parser.add_argument('--checkpoint_path', type=str, default=None)
    parser.add_argument('--checkpoint_interval', type=int, default=1)
    parser.add_argument('--save_checkpoint', type=str2bool, nargs='?', const=True, default=True)
    parser.add_argument('--seed', type=int, default=None,
                        help='Seed Python/NumPy/torch(+CUDA) and the DataLoader for reproducible '
                             'training. Default None = original unseeded behaviour.')

    args = parser.parse_args()
    diffusion_train_pm(
        y_emb_path=args.y_emb_path,
        emb_alignment_path=args.emb_alignment_path,
        diffusion_path=args.diffusion_path,
        y_type=args.y_type,
        use_tab=args.use_tab,
        use_text=args.use_text,
        use_image=args.use_image,
        vae_dat_path=args.vae_dat_path,
        text_path=args.text_path,
        image_path=args.image_path,
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        d_in=args.d_in,
        dim_t=args.dim_t,
        cond_dim=args.cond_dim,
        text_pooling=args.text_pooling,
        normalize_tab_cond=args.normalize_tab_cond,
        normalize_text_tokens=args.normalize_text_tokens,
        normalize_text_pooled=args.normalize_text_pooled,
        normalize_projected_cond=args.normalize_projected_cond,
        alignment_mix_alpha=args.alignment_mix_alpha,
        raw_vae_dat_path=args.raw_vae_dat_path,
        raw_text_path=args.raw_text_path,
        early_stopping_thrshd=args.early_stopping_thrshd,
        clip_grad=args.clip_grad,
        clip_grad_max_norm=args.clip_grad_max_norm,
        diffusion_class_weights=args.diffusion_class_weights,
        normalize_diffusion_class_weights=args.normalize_diffusion_class_weights,
        save_milestone_epochs=args.save_milestone_epochs,
        use_wandb=args.use_wandb,
        wandb_project=args.wandb_project,
        run_group=args.run_group,
        run_name=args.run_name,
        wandb_log_interval=args.wandb_log_interval,
        resume=args.resume,
        checkpoint_path=args.checkpoint_path,
        checkpoint_interval=args.checkpoint_interval,
        save_checkpoint=args.save_checkpoint,
        seed=args.seed,
    )

if __name__ == '__main__':
    main()
