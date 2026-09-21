import json
from safetensors.torch import load_file, save_file

snap = '/snap/snapshots/904f815deed0fbeab02d82d4648b70106272ffe3'
idx = json.load(open(f'{snap}/model.safetensors.index.json'))
wm = idx['weight_map']
mtp = {k for k in wm if 'mtp' in k.lower()}
files = sorted(set(wm.values()))
new_map = {}
for f in files:
    tensors = load_file(f'{snap}/{f}')
    kept = {k: v for k, v in tensors.items() if k not in mtp}
    save_file(kept, f'/out/{f}', metadata={'format': 'pt'})
    print(f, len(tensors), '->', len(kept))
    for k in kept:
        new_map[k] = f

idx['weight_map'] = new_map
json.dump(idx, open('/out/model.safetensors.index.json', 'w'), indent=2)
print('mtp stripped:', len(mtp), 'keys')
