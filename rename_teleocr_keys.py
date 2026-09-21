from safetensors.torch import load_file, save_file
import shutil

p = '/m/model.safetensors'
tensors = load_file(p)
renamed = {}
for k, v in tensors.items():
    if k == 'lm_head.weight':
        nk = 'model.lm_head.weight'
    elif k.startswith('model.layers.'):
        nk = 'model.model.language_model.' + k[len('model.'):]
    elif k == 'model.embed_tokens.weight':
        nk = 'model.model.language_model.embed_tokens.weight'
    elif k == 'model.norm.weight':
        nk = 'model.model.language_model.norm.weight'
    elif k.startswith('visual.'):
        nk = 'model.model.' + k
    else:
        raise SystemExit('unmapped key: ' + k)
    renamed[nk] = v

shutil.copy(p, '/m/model.safetensors.origkeys')
save_file(renamed, p, metadata={'format': 'pt'})
print('renamed', len(renamed), 'keys')
