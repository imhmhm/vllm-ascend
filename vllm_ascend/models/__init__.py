from vllm import ModelRegistry

import vllm_ascend.envs as envs_ascend


def register_model():
    ModelRegistry.register_model("DeepseekV4ForCausalLM", "vllm_ascend.models.deepseek_v4:AscendDeepseekV4ForCausalLM")

    ModelRegistry.register_model("DeepSeekV4MTPModel", "vllm_ascend.models.deepseek_v4_mtp:DeepSeekV4MTP")
    ModelRegistry.register_model(
        "LlamaForCausalLMVwnEagle3", "vllm_ascend.models.llama_eagle3_vwn:Eagle3VwnLlamaForCausalLM"
    )
    ## ailab_slm patch
    if envs_ascend.AILAB_SLM_LOCAL_PATH is not None:
        from .ailab_slm import AILabSLMForCausalLM
        ModelRegistry.register_model(
            "AILabSLMForCausalLM",
            "vllm_ascend.models.ailab_slm:AILabSLMForCausalLM"
        )
