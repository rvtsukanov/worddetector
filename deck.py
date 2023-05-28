class Node:
    def __init__(self):
        self.children = {}
        self.word = None


class Trie:
    def __init__(self):
        self.root = Node()

    def insert(self, word: str) -> None:
        node = self.root
        for char in word:
            if char not in node.children:
                node.children[char] = Node()
            node = node.children[char]

        node.word = word

    def search(self, word: str) -> bool:
        node = self.root
        for char in word:
            if char in node.children:
                node = node.children[char]
            else:
                return False
        if node.word == word:
            return True
        else:
            return False

    def startsWith(self, prefix: str) -> bool:
        node = self.root
        for char in prefix:
            if char in node.children:
                node = node.children[char]
            else:
                return False

        return True

words = ['ель', 'нож', 'кит', 'енот', 'кон', 'кол', 'тон']

trie = Trie()
for word in words:
    trie.insert(word)

deck = [['е', 'л', 'ь'],
        ['н', 'о',  'ж'],
        ['к', 'и', 'т']]


def proceed(deck, current_prefix, row, col, parent_trie):
    char = deck[row][col]
    word_flag = parent_trie.search(current_prefix)


    # trie = parent_trie.root.children[char]






def find_words(deck, trie):
    nrows = len(deck)
    ncols = len(deck[0])

    for row in range(nrows):
        for col in range(ncols):
            if deck[row][col] in trie.root.children:
                current_prefix = deck[row][col]
                proceed(deck, current_prefix, row, col, trie)