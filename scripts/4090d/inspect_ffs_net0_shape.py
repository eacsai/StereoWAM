import torch, sys
sys.path.insert(0, '/data/wangqiwei/ICLR2026/Fast-FoundationStereo')
ffs = torch.load('/data/wangqiwei/ICLR2026/Fast-FoundationStereo/weights/20-30-48/model_best_bp2_serialize.pth', map_location='cuda', weights_only=False)
ffs.eval()
print('hidden_dims =', ffs.args.hidden_dims, '| valid_iters =', ffs.args.valid_iters, '| n_gru_layers =', ffs.args.get('n_gru_layers', '?'))
cap = {}
def hook(m, i, o):
    cap['out_type'] = type(o).__name__
    cap['out_len'] = len(o) if isinstance(o, (tuple, list)) else 'NA'
    net = o[0]
    cap['net_type'] = type(net).__name__
    cap['net_len'] = len(net) if isinstance(net, (tuple, list)) else 'NA'
    cap['net0_shape'] = tuple(net[0].shape)
ffs.update_block.register_forward_hook(hook)
img1 = (torch.rand(1,3,256,256, device='cuda')*255)
img2 = (torch.rand(1,3,256,256, device='cuda')*255)
with torch.no_grad():
    ffs(img1, img2, iters=int(ffs.args.valid_iters), test_mode=True)
print('update_block output:', cap['out_type'], 'len', cap['out_len'])
print('net (output[0]):', cap['net_type'], 'len', cap['net_len'])
print('net[0] shape =', cap['net0_shape'])
