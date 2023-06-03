from worddetector.deck_leetcoded import findWords
import pytest

@pytest.fixture
def words_list():
    words = []
    with open('./corpus/normalized_corpus_rus.txt', 'r') as f:
        for line in f.readlines():
            words.append(line.strip())
    return words


@pytest.fixture
def deck_3x3():
    deck = [['е', 'л', 'ь'],
            ['н', 'о',  'ж'],
            ['к', 'и', 'т']]

    return deck


def test_corpus(words_list):
    assert len(words_list) == 162507


def test_3x3_deck(deck_3x3, words_list):
    answer = findWords(deck_3x3, words_list)
    assert len(answer) == 49
