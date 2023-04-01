import nltk
import tqdm
from nltk.tree import ParentedTree
import SBAR
from nltk.parse import CoreNLPParser
import re
import os
from anytree import AnyNode

# os.environ['CLASSPATH']="F:\\Anaconda3\\NLP\\stanford-parser-full-2018-02-27;C:\\Users\\Lokesh\\AppData\\Roaming\\nltk_data\\taggers\\averaged_perceptron_tagger;F:\\Anaconda3\\NLP\\stanford-ner-2015-12-09"
# For nltk tagger

parser = CoreNLPParser(url='http://localhost:9000/')

split = []
simple_sent = []
index = []
index1 = 0
n = 0
but = 0
scount = 0
parts = []
ht_3_last_obj = []

# SBAR functions start here
def make_tree_sbar(tree, t, sent_list):

    # this fn. converts nltk tree to anytree

    if tree not in sent_list:
        ttt = AnyNode(id=str(tree.label()), parent=t)
        for tt in tree:
            make_tree_sbar(tt, ttt, sent_list)
    else:
        AnyNode(id=str(tree), parent=t)


def find_sbar(t):
    if t.id == 'SBAR':
        global sbar
        sbar = t
    for tt in t.children:
        find_sbar(tt)


def find_vp_in_sbar(t):
    if t.id == 'VP':
        global vp_sbar
        vp_sbar = t
    for tt in t.children:
        find_vp_in_sbar(tt)


def find_vp(t):
    if t.id == 'SBAR':
        return
    global f
    if t.id == 'VP' and f == True:
        global vp
        vp = t
        f = False
    for tt in t.children:
        find_vp(tt)


def find_np(t):
    if t.id == 'SBAR':
        return
    global f
    if t.id == 'NP' and f == True:
        global np
        np = t
        f = False
    for tt in t.children:
        find_np(tt)


def find_vbz(t):
    if t.id == 'SBAR':
        return
    global f
    if t.id == 'VBZ' and f == True:
        global vbz
        vbz = t.children[0].id
        f = False
    for tt in t.children:
        find_vbz(tt)


def make_sent(t):
    global simple_sentences
    if t.id in sent_list:
        simple_sentences[-1].append(t.id)
    for tt in t.children:
        make_sent(tt)


# SBAR functions end here

# Multiple CC functions start here
def pos_tag(tokenized_sent):
    return nltk.pos_tag(tokenized_sent)


def has_conj(tagged_sent):
    cc_list = [('and', 'CC'), ('but', 'CC')]
    for cc_pair in cc_list:
        if cc_pair in tagged_sent:
            return True
    return False


def split_needed(sent_list):
    for sent in sent_list:
        if has_conj(pos_tag(tokenize(sent))):
            return True
    return False


def split(sent, cc_tuple):
    pos_tagged = pos_tag(tokenize(sent))
    tree = next(parser.parse(tokenize(sent)))
    tree1 = ParentedTree.convert(tree)

    count = 0
    m = 0
    for t in tree1.subtrees():
        if t.label() == 'PP':
            count = count + 1

    index = []
    index1 = 0
    if count > 0 and 'to' not in tokenized_sent and 'washed' \
            not in tokenized_sent and tokenized_sent.count(r",") < 2:
        for i in range(len(pos_tagged) - 3):
            if (pos_tagged[i][1] == 'VBD' or pos_tagged[i][1] == 'VBZ') \
                    and pos_tagged[i + 1][1] != 'VBG' and pos_tagged[i + 3][1] != 'CC' and pos_tagged[i + 1][1] != 'NNP' \
                    and pos_tagged[i - 1][1] != 'CC':
                pos_tagged.insert(i + 1, (r",", r","))

        for j in range(len(pos_tagged)):
            if pos_tagged[j][1] == 'CC':
                index.append(j)

    for t in tree1.subtrees():
        if t.label() == 'SBAR':
            m = m + 1
    if len(index) > 0 and count > 0 and m == 0:
        c = 0
        for i in range(len(index)):
            pos_tagged.insert(index[i] + c, (r",", r","))
            c = c + 1
    if m > 0:
        for j in range(len(pos_tagged)):
            if pos_tagged[j][1] == 'CC':
                index1 = j

    if index1 > 0 and m > 0 and count == 0:
        pos_tagged.insert(index1, (' ,', r","))  # ', 'is used
        pos_tagged.insert(index1 + 2, (', ', r","))  # ' ,' is used

    tree = next(parser.parse(tokenized_sent))
    p_tree = ParentedTree.convert(tree)

    leaf_values = p_tree.leaves()
    parts = []
    ht_3_last_obj = []

    if cc_tuple in pos_tagged:
        leaf_index = leaf_values.index(cc_tuple[0])
        tree_location = p_tree.leaf_treeposition(leaf_index)
        parent = p_tree[tree_location[:-2]]

        if parent.height() == 3:

            # find the noun being referred to
            for subtree in reversed(list(parent.subtrees())):
                if subtree.parent() == parent:
                    if subtree.label() == 'NN' or subtree.label() \
                            == 'NNS':
                        ht_3_last_obj = subtree.leaves() + ht_3_last_obj
                        del p_tree[subtree.treeposition()]

            part = []
            for subtree in reversed(list(parent.subtrees())):
                if subtree.parent() == parent:

                    if subtree.label() != r"," and subtree.label() \
                            != 'CC':
                        part = subtree.leaves() + part
                    else:
                        parts.append(part + ht_3_last_obj)
                        part = []
                    del p_tree[subtree.treeposition()]
            parts.append(part + ht_3_last_obj)

            parent.append(ParentedTree('INSRT', ['*']))
        else:

            for subtree in reversed(list(parent.subtrees())):
                if subtree.parent() == parent:

                    if subtree.label() != r"," and subtree.label() \
                            != 'CC':
                        parts.append(subtree.leaves() + ht_3_last_obj)
                    del p_tree[subtree.treeposition()]

            parent.append(ParentedTree('INSRT', ['*']))

    split = []
    rem = p_tree.leaves()
    start_idx = rem.index('*')

    for part in reversed(parts):
        offset = start_idx
        r_clone = rem.copy()
        del r_clone[offset]
        for (i, word) in enumerate(part):
            r_clone.insert(offset + i, word)
        split.append(r_clone)

    split = [' '.join(sent) for sent in split]

    return split


def split_util(sent):
    cc_list = [('and', 'CC'), ('but', 'CC')]
    for cc_pair in cc_list:
        if cc_pair in pos_tag(tokenize(sent)):
            return split(sent, cc_pair)
    return sent


def rem_dup(list):
    final = []
    for item in list:
        if item not in final:
            final.append(item)
    return final


def simplify(sent):
    initial = [sent]
    final = []

    # Recursively remove the conjunctions
    while split_needed(initial):
        final = []
        while initial:
            sent = initial.pop(0)
            if split_needed([sent]):
                for split_sent in reversed(split_util(sent)):
                    final.append(split_sent)
            else:
                final.append(sent)

        initial = final.copy()

    final = rem_dup(final)
    final = list(reversed(final))

    return final


def tokenize(sent):
    sent = str(sent)
    tokenized_sent = nltk.word_tokenize(sent)
    if 'If' in tokenized_sent and 'then' in tokenized_sent:
        tokenized_sent.remove('If')
        tokenized_sent.insert(tokenized_sent.index('then'), 'and')
        tokenized_sent.remove('then')
    if 'because' in tokenized_sent:
        tokenized_sent.insert(tokenized_sent.index(
            'because'), r",")  # ', 'is used
        tokenized_sent.insert(tokenized_sent.index('because') + 1, r",")
        tokenized_sent.insert(tokenized_sent.index('because'), 'and')
        tokenized_sent.remove('because')
    if 'while' in tokenized_sent:
        tokenized_sent.insert(tokenized_sent.index('while'), 'and')
        tokenized_sent.remove('while')
    if 'which' in tokenized_sent:
        tokenized_sent.insert(tokenized_sent.index('which'), 'and')
        tokenized_sent.remove('which')
    if 'or' in tokenized_sent:
        tokenized_sent.insert(tokenized_sent.index('or'), 'and')
        tokenized_sent.remove('or')
    if 'who' in tokenized_sent:
        while r"," in tokenized_sent:
            tokenized_sent.insert(tokenized_sent.index(r","), 'and')
            tokenized_sent.remove(r",")
        tokenized_sent.insert(tokenized_sent.index('who'), 'and')
        tokenized_sent.remove('who')

    return tokenized_sent

tokenized_sent = ''
sent_list = ''
def serverParser(sentence):
    global tokenized_sent
    global sent_list
    global n
    global but
    # print(sentences.index(sentence))
    # print("ComplexSentence: "+sentence)
    tokenized_sent = tokenize(sentence)
    # print(tokenized_sent)
    pos_tagged = pos_tag(tokenized_sent)
    parse_trees = parser.parse(tokenized_sent)
    tree = next(parse_trees)
    p_tree = ParentedTree.convert(tree)
    leaf_values = p_tree.leaves()
    for i in pos_tagged:
        if 'and' in i:
            n = n + 1

        if 'but' in i:
            but = but + 1
    tree1 = ParentedTree.convert(tree)
    m = 0
    for t in tree1.subtrees():
        if t.label() == 'SBAR':
            m = m + 1

    simple_sentence = ''
    if n + but > 0:
        # tokenized_sent=nltk.word_tokenize(sent10)
        # pos_tagged=nltk.pos_tag(tokenized_sent)
        sent1 = sentence
        sent = ' '.join(tokenize(sent1))
        simplified = simplify(sent)
        for i in simplified:
            i = list(i)
            if ord(i[0]) >= 97 and ord(i[0]) <= 122:
                i[0] = chr(ord(i[0]) - 32)
            while i.count(r",") > 0:
                del i[i.index(r",")]
            if '.' not in i:
                # print("Simple sentence: "+"".join(i)+".", '\n\n')
                simple_sentence += ''.join(i) + '. '
            else:
                # print("Simple sentence: "+"".join(i), '\n\n')
                simple_sentence += ''.join(i) + ' '
        n = 0
        but = 0
    elif n == 0 and m > 0 and len(re.findall(r",", sentence)) == 0 \
            and len(re.findall(r"While", sentence)) == 0:
        # print("."),
        try:
            sent = sentence
            # print(sent)
            # print("Hello")
            tokenized_sent = tokenize(sent)
            pos_tagged = nltk.pos_tag(tokenized_sent)
            parse_trees = parser.parse(tokenized_sent)
            sent_list = [s for s in sent.split()]
            tree = next(parse_trees)[0]
            # tree.draw()
            t = AnyNode(id='ROOT')
            make_tree_sbar(tree, t, sent_list)
            sbar = t
            vp_sbar = t
            vp = t
            np = t
            vbz = 'asvf'
            find_sbar(t)
            find_vp_in_sbar(sbar)
            f = True
            find_vp(t)
            f = True
            find_np(t)
            f = True
            find_vbz(t)
            simple_sentences = []
            simple_sentences.append([])
            make_sent(np)
            make_sent(vp)
            simple_sentences.append([])
            make_sent(np)
            if vbz != 'asvf':
                simple_sentences[-1].append(vbz)
            make_sent(vp_sbar)
            for i in simple_sentences:
                i = list(i)
                while i.count(r",") > 0:
                    i.pop(i.index(r","))
                if '.' not in i:
                    # print("Simple sentence: "+" ".join(i)+".", '\n\n')
                    simple_sentence += ' '.join(i) + '.'
                else:
                    # print("Simple sentence: "+" ".join(i), '\n\n')
                    simple_sentence += ' '.join(i)
        except:
            # print("."),
            simple_sentence += sentence
    elif m > 0 and (len(re.findall(r",", sentence)) > 0
                    or len(re.findall(r"While", sentence)) > 0):
        try:
            # sent=re.sub(r",","",sentence)
            # print("Hello")
            tokenized_sent = tokenize(sentence)
            simple_sentences = \
                SBAR.simplify(' '.join(tokenized_sent))
            for i in simple_sentences:
                if '.' not in i:
                    # print("Simple sentence: "+i, '\n\n')
                    simple_sentence += i + ' '
                else:
                    # print("Simple sentence: "+i, '\n\n')
                    simple_sentence += i + ' '
        except:
            simple_sentence += sentence
    else:
        simple_sentence += sentence

    simple_sentence = ''.join(simple_sentence).replace('\n', '')
    return simple_sentence