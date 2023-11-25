import argparse
import torch
from utils import remap_checkpoint_keys
from utils import remap_checkpoint_keys_r50

def main(args):
    checkpoint = torch.load(args.input_dir, map_location='cpu')
    print("Load pre-trained checkpoint from: %s" % args.input_dir)
    checkpoint_model = checkpoint['online_without_ddp']
    # remove decoder weights
    checkpoint_model_keys = list(checkpoint_model.keys())
    for k in checkpoint_model_keys:
        if 'decoder' in k or 'mask_token'in k or \
            'proj' in k or 'pred' in k:
            print(f"Removing key {k} from pretrained checkpoint")
            del checkpoint_model[k]
    
    try:
        checkpoint_model = remap_checkpoint_keys(checkpoint_model)
    except RuntimeError:
        checkpoint_model = remap_checkpoint_keys_r50(checkpoint_model)
    torch.save(checkpoint_model, args.output_dir)


if __name__ == '__main__':
    parser = argparse.ArgumentParser('Self-supervised weights', add_help=False)
    parser.add_argument('--input_dir', default='./pretrain/checkpoint-99.pth',
                        help='pretrain checkpoing')
    parser.add_argument('--output_dir', default='./downstream/VOS/checkpoints/mac-small.pth',
                        help='path where to save, empty for no saving')
    args = parser.parse_args()
    main(args)