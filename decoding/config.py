device_map = {
    'model.decoder.embed_tokens': 3,
    'model.decoder.embed_positions': 2,
}

# 레이어를 절반씩 배치
for i in range(32):
    device_map[f'model.decoder.layers.{i}'] = 2 if i < 20 else 3

device_map.update({
    'model.decoder.final_layer_norm': 3,
    'lm_head': 3,
})