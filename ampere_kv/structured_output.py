"""XGrammar可选适配：编译结果可复用，请求独占matcher和掩码。"""

from dataclasses import dataclass

import torch


@dataclass(frozen=True)
class CompiledSchema:
    grammar: object
    vocab_size: int
    stop_token_ids: tuple[int, ...]
    valid_bits: torch.Tensor

    def new_state(self, device):
        return StructuredOutput(self, device)


class SchemaCompiler:
    def __init__(self, tokenizer, vocab_size, stop_token_ids):
        import xgrammar as xgr

        self.vocab_size = vocab_size
        self.stop_token_ids = tuple(sorted(stop_token_ids))
        tokens = set(tokenizer.get_vocab().values())
        if not self.stop_token_ids or any(t not in tokens or t >= vocab_size for t in self.stop_token_ids):
            raise ValueError("停止Token必须来自实际tokenizer并在模型词表内")
        info = xgr.TokenizerInfo.from_huggingface(tokenizer, vocab_size=vocab_size,
                                                stop_token_ids=list(self.stop_token_ids))
        self.compiler = xgr.GrammarCompiler(info, max_threads=1, cache_enabled=False)
        # 明确排除补齐词表与编号空洞，不把它们当成可生成的空字符串。
        words = [0] * ((vocab_size + 31) // 32)
        for token in tokens:
            if 0 <= token < vocab_size:
                words[token // 32] |= 1 << (token % 32)
        self.valid_bits = torch.tensor([[w if w < 2**31 else w - 2**32 for w in words]], dtype=torch.int32)

    def compile(self, schema):
        grammar = self.compiler.compile_json_schema(schema, any_whitespace=False,
                                                     separators=(",", ":"), strict_mode=True)
        return CompiledSchema(grammar, self.vocab_size, self.stop_token_ids, self.valid_bits)


class StructuredOutput:
    def __init__(self, compiled, device):
        import xgrammar as xgr

        self.xgr, self.compiled = xgr, compiled
        self.matcher = xgr.GrammarMatcher(compiled.grammar)
        self.cpu_mask = xgr.allocate_token_bitmask(1, compiled.vocab_size)
        self.device_mask = self.cpu_mask.to(device)

    def mask(self, logits):
        self.matcher.fill_next_token_bitmask(self.cpu_mask)
        self.cpu_mask.bitwise_and_(self.compiled.valid_bits)
        if not self.cpu_mask.any():
            raise RuntimeError("结构化语法没有合法Token，停止生成")
        self.device_mask.copy_(self.cpu_mask)
        self.xgr.apply_token_bitmask_inplace(logits.view(1, -1), self.device_mask,
                                           vocab_size=self.compiled.vocab_size)

    def accept(self, token):
        if not self.matcher.accept_token(token):
            raise RuntimeError(f"提交Token {token}不符合请求语法")
        return self.matcher.is_terminated()

    def release(self):
        self.matcher = self.compiled = self.xgr = None
        self.cpu_mask = self.device_mask = None
