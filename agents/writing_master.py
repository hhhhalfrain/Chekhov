from llm_base import LLMBase

class WritingMaster(LLMBase):
    def __init__(self, env_path: str):
        super().__init__(env_path)


    SYSTEM = """你是一名资深的写作指导专家，专注于提升各类写作作品的质量，包括小说、剧本、散文等。你的任务是根据用户提供的写作内容，给予具体且实用的建议
不要包含任何与写作无关的内容，不用提示用户追问，仅思考。
"""
    def provide_guidance(self,prompt: str) -> str:
        response = self.call_structured_json(
            model=self.STRONG_TEXT_MODEL,
            system_prompt=self.SYSTEM,
            user_prompt=prompt,
            json_schema={
                "name": "guidance_response",
                "schema": {
                    "type": "object",
                    "properties": {
                        "guidance": {
                            "type": "string",
                            "description": "High-quality writing guidance and suggestions.",
                        },
                    },
                    "required": ["guidance"],
                },
            },
            temperature=0.7,
        )
        return response["guidance"]