device_map = {
    "model.decoder.embed_tokens": 0,      
    "model.decoder.embed_positions": 0,
    "model.decoder.final_layer_norm": 0,
    "lm_head": 0,
}

# 레이어 분산 설정 (0~19: cuda:0, 20~39: cuda:2)
for i in range(40):
    if i < 20:
        device_map[f"model.decoder.layers.{i}"] = 0
    else:
        device_map[f"model.decoder.layers.{i}"] = 1
