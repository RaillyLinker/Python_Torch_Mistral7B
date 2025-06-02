import torch
from transformers import GPT2TokenizerFast
import torch.nn.functional as F


def generate_text(model, tokenizer, device, start_prompt="", max_new_tokens=1000, temperature=1.0):
    model.eval()
    generated = []

    # 시작 토큰 시퀀스 생성
    if start_prompt:
        input_ids = tokenizer.encode(start_prompt, return_tensors="pt").to(device)
    else:
        # 시작 문구 없으면 임의 토큰 하나 생성 (예: <BOS> 또는 eos_token)
        input_ids = torch.tensor(
            [[tokenizer.bos_token_id if tokenizer.bos_token_id is not None else tokenizer.eos_token_id]], device=device)

    with torch.no_grad():
        for _ in range(max_new_tokens):
            # 입력 시퀀스가 너무 길면 뒤쪽 일부만 자르기 (모델 최대 길이 내로)
            if input_ids.size(1) > model.max_len:
                input_ids = input_ids[:, -model.max_len:]

            logits = model(input_ids)  # (batch=1, seq_len, vocab_size)
            logits = logits[:, -1, :] / temperature  # 마지막 토큰의 로짓만 사용 (1, vocab_size)
            probs = F.softmax(logits, dim=-1)

            # 샘플링: 확률 분포에서 다음 토큰 선택
            next_token_id = torch.multinomial(probs, num_samples=1)

            # 시퀀스에 다음 토큰 추가
            input_ids = torch.cat([input_ids, next_token_id], dim=-1)

            generated.append(next_token_id.item())

    generated_text = tokenizer.decode(generated, clean_up_tokenization_spaces=True)
    return generated_text


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # 토크나이저 로드 (학습 시 사용한 것과 동일)
    tokenizer = GPT2TokenizerFast.from_pretrained("gpt2")
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # 모델 로드 (학습 시 저장한 경로)
    model = torch.load("mistral7b_toy.pth", map_location=device, weights_only=False)
    model.to(device)

    # 시작 문구 설정 (빈 문자열 가능)
    start_prompt = "The meaning of life is"

    generated_text = generate_text(model, tokenizer, device, start_prompt=start_prompt, max_new_tokens=1000)
    print("Generated Text:\n")
    print(generated_text)


if __name__ == "__main__":
    main()
