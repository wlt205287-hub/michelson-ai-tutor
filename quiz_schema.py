from __future__ import annotations

from typing import Annotated, Literal
from uuid import uuid4

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator


class QuizQuestionBase(BaseModel):
    model_config = ConfigDict(extra="forbid", str_strip_whitespace=True)

    id: str = Field(min_length=1)
    question: str = Field(min_length=1)


class ChoiceQuestion(QuizQuestionBase):
    type: Literal["choice"]
    options: list[str] = Field(min_length=4, max_length=4)
    correct_answer: int = Field(ge=0, le=3)
    explanation: str = Field(min_length=1)

    @field_validator("options")
    @classmethod
    def validate_options(cls, options: list[str]) -> list[str]:
        cleaned = [option.strip() for option in options]
        if any(not option for option in cleaned):
            raise ValueError("选择题选项不能为空")
        if len(set(cleaned)) != len(cleaned):
            raise ValueError("选择题选项不能重复")
        return cleaned


class ShortAnswerQuestion(QuizQuestionBase):
    type: Literal["short_answer"]
    reference_answer: str = Field(min_length=1)
    key_points: list[str] = Field(min_length=3, max_length=5)

    @field_validator("key_points")
    @classmethod
    def validate_key_points(cls, key_points: list[str]) -> list[str]:
        cleaned = [point.strip() for point in key_points]
        if any(not point for point in cleaned):
            raise ValueError("简答题评分要点不能为空")
        if len(set(cleaned)) != len(cleaned):
            raise ValueError("简答题评分要点不能重复")
        return cleaned


QuizQuestion = Annotated[
    ChoiceQuestion | ShortAnswerQuestion,
    Field(discriminator="type"),
]


class GeneratedQuiz(BaseModel):
    model_config = ConfigDict(extra="forbid", str_strip_whitespace=True)

    topic: str = Field(min_length=1)
    questions: list[QuizQuestion] = Field(min_length=5, max_length=5)

    @model_validator(mode="after")
    def validate_questions(self) -> "GeneratedQuiz":
        question_ids = [question.id for question in self.questions]
        if len(set(question_ids)) != len(question_ids):
            raise ValueError("题目 id 必须唯一")

        counts = {"choice": 0, "short_answer": 0}
        for question in self.questions:
            counts[question.type] += 1
        if counts != {"choice": 3, "short_answer": 2}:
            raise ValueError("题型分布必须为3道选择题、2道简答题")
        return self


class GeneratedQuizEnvelope(BaseModel):
    model_config = ConfigDict(extra="forbid")

    quiz: GeneratedQuiz


class Quiz(GeneratedQuiz):
    quiz_id: str = Field(min_length=1)
    stage: Literal["pre_lab"] = "pre_lab"


class QuizEnvelope(BaseModel):
    model_config = ConfigDict(extra="forbid")

    quiz: Quiz


def build_quiz_envelope(generated: GeneratedQuizEnvelope) -> QuizEnvelope:
    return QuizEnvelope(
        quiz=Quiz(
            quiz_id=f"quiz_{uuid4().hex}",
            stage="pre_lab",
            **generated.quiz.model_dump(),
        )
    )
