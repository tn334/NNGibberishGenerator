
# classes to handle creation of frequency tree
class TreeNode:
    def __init__(self) -> None:
        self.letter = ""
        self.frequencyCount = 0
        self.children = dict()
        self.endOfWord = False

class FrequencyTree:
    def __init__(self) -> None:
        self.root = TreeNode() # set root of tree
    
    def insertWord(self, word):
        currentNode = self.root # wkg pointer for root

        # loop through letters in word
        for char in word:

            # check if letter is not a child of current node
            if char not in currentNode.children:
                currentNode.children[char] = TreeNode() # add child node
            
            # move to next letter (node)
            currentNode = currentNode.children[char]
        currentNode.endOfWord = True

    def printTree(self, node=None, word=''):
        if node is None:
            node = self.root
        
        if node.endOfWord:
            print(word)
        
        for char, childNode in node.children.items():
            self.printTree(childNode, word + char)




def main():
    # read all words in txt file
    with open('test.txt', 'r') as file:
        words = file.readlines()

    tree = FrequencyTree()

    for word in words:
        word = word.strip()
        tree.insertWord(word)
    
    tree.printTree()




if __name__ == "__main__":
    main()






############################   OLD CODE THAT CAN BE USEFUL   ###################################   
#frequencyDict = { letter: {} for letter in 'abcdefghijklmnopqrstuvwxyz'}

# # loop through words
# for word in words:
#     word = word.strip()

#     # compare position to letter and count positions
#     for position, letter in enumerate(word):
#         if position not in frequencyDict[letter]:
#             frequencyDict[letter][position] = 1
#         else:
#             frequencyDict[letter][position] += 1

# # write freq dict contents to file
# with open('letter_frequency.txt', 'w') as file:
#     for letter, positions in frequencyDict.items():
#         file.write(f"{letter}: {positions}\n\n")




