def words_list(path='../../corpus/normalized_corpus_rus.txt'):
    words = []
    with open(path, 'r') as f:
        for line in f.readlines():
            words.append(line.strip())
    return words


def findWords(board, words):
    for i in range(len(board)):
        for j in range(len(board[0])):
            board[i][j] = board[i][j].lower()

    WORD_KEY = '$'

    trie = {}
    for word in words:
        node = trie
        for letter in word:
            # retrieve the next node; If not found, create a empty node.
            node = node.setdefault(letter, {})
        # mark the existence of a word in trie node
        node[WORD_KEY] = word

    rowNum = len(board)
    colNum = len(board[0])

    matchedWords = []

    def backtracking(row, col, parent):

        letter = board[row][col]
        currNode = parent[letter]

        # check if we find a match of word
        word_match = currNode.pop(WORD_KEY, False)
        if word_match:
            # also we removed the matched word to avoid duplicates,
            #   as well as avoiding using set() for results.
            matchedWords.append(word_match)

        # Before the EXPLORATION, mark the cell as visited
        board[row][col] = '#'

        # Explore the neighbors in 4 directions, i.e. up, right, down, left
        for (rowOffset, colOffset) in [(-1, 0), (0, 1), (1, 0), (0, -1), (1, 1), (-1, -1), (1, -1), (-1, 1)]:
            newRow, newCol = row + rowOffset, col + colOffset
            if newRow < 0 or newRow >= rowNum or newCol < 0 or newCol >= colNum:
                continue
            if not board[newRow][newCol] in currNode:
                continue
            backtracking(newRow, newCol, currNode)

        # End of EXPLORATION, we restore the cell
        board[row][col] = letter

        # Optimization: incrementally remove the matched leaf node in Trie.
        if not currNode:
            parent.pop(letter)

    for row in range(rowNum):
        for col in range(colNum):
            # starting from each of the cells
            if board[row][col] in trie:
                backtracking(row, col, trie)

    return matchedWords

# words = ['ель', 'нож', 'кит', 'енот', 'кон', 'кол', 'тон']

# deck = [['е', 'л', 'ь'],
#         ['н', 'о',  'ж'],
#         ['к', 'и', 'т']]


# print(findWords(deck, words))