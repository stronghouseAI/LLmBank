"""The yes/no questions and input rules behind `shai agentvault new`."""

import pytest

from stronghouse.core.agent_builder import (
    SKILL_QUESTIONS,
    creativity_for_temperature,
    diligence_for_max_steps,
    parse_skill_picks,
    parse_yes_no,
    skills_from_answers,
)
from stronghouse.core.agents import max_steps_for_diligence, temperature_for_creativity
from stronghouse.core.skill_catalog import SKILLS


def test_all_no_gives_an_agent_with_no_skills():
    assert skills_from_answers([False] * len(SKILL_QUESTIONS)) == []


def test_each_yes_adds_only_its_own_skills_in_catalog_order():
    answers = [False] * len(SKILL_QUESTIONS)
    internet = next(i for i, q in enumerate(SKILL_QUESTIONS) if "internet" in q.text)
    answers[internet] = True
    assert skills_from_answers(answers) == ["fetch_web_page", "read_news_feed", "web_search"]


def test_every_skill_can_be_reached_by_a_question():
    all_yes = skills_from_answers([True] * len(SKILL_QUESTIONS))
    assert all_yes == list(SKILLS)


def test_answers_must_match_the_questions():
    with pytest.raises(ValueError):
        skills_from_answers([True])


@pytest.mark.parametrize(
    ("text", "default", "expected"),
    [("y", False, True), ("YES", False, True), ("n", True, False), ("", True, True),
     ("", False, False), ("maybe", True, None)],
)  # fmt: skip
def test_yes_no_answers(text, default, expected):
    assert parse_yes_no(text, default) is expected


def test_skill_picks_take_numbers_and_names_without_repeats():
    offered = list(SKILLS)
    assert parse_skill_picks("1, web_search 1", offered) == [offered[0], "web_search"]
    with pytest.raises(ValueError, match="not a skill"):
        parse_skill_picks("teleport", offered)
    with pytest.raises(ValueError):
        parse_skill_picks("99", offered)


def test_temperature_and_max_steps_round_trip_to_stats():
    for stat in range(1, 11):
        assert creativity_for_temperature(temperature_for_creativity(stat)) == stat
        assert diligence_for_max_steps(max_steps_for_diligence(stat)) == stat
    assert diligence_for_max_steps(11) == 4  # odd numbers round down to a real setting
    with pytest.raises(ValueError):
        creativity_for_temperature(1.5)
    with pytest.raises(ValueError):
        diligence_for_max_steps(2)
