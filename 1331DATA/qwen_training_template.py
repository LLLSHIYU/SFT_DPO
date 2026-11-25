"""Enable assistant-only loss for ordinary assistant turns in the pinned Qwen template."""


def enable_assistant_loss_mask(tokenizer):
    """Add assistant span markers to a dedicated SFT tokenizer instance in place."""
    original = r"""{{- '<|im_start|>' + message.role + '\n' + message.content + '<|im_end|>' + '\n' }}"""
    replacement = r"""{%- if message.role == "assistant" %}
        {{- '<|im_start|>assistant\n' }}
        {%- generation %}{{- message.content + '<|im_end|>\n' }}{%- endgeneration %}
    {%- else %}
        """ + original + """
    {%- endif %}"""
    template = tokenizer.chat_template
    if not isinstance(template, str) or template.count(original) != 1:
        raise ValueError("Expected the pinned Qwen2.5 ChatML template.")
    tokenizer.chat_template = template.replace(original, replacement, 1)
