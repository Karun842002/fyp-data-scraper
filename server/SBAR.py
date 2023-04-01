def simplify(sent):
    from anytree import NodeMixin, Node, AnyNode, RenderTree

    def make_tree(tree, t, sent_list):
        # this fn. converts nltk tree to anytree
        if tree not in sent_list:
            ttt = AnyNode(id=str(tree.label()), parent=t)
            for tt in tree:
                make_tree(tt, ttt, sent_list)
        else:
            AnyNode(id=str(tree), parent=t)

    parser = CoreNLPParser(url='http://localhost:9000/')
    from nltk.parse import CoreNLPParser

    def find_sbar(t):
        if t.id == 'SBAR':
            global sbar
            sbar = t
        for tt in t.children:
            find_sbar(tt)

    def find_vp_in_sbar(t):
        if t.id == 'VP':
            global vp_sbar
            vp_sbar.append(t)
        for tt in t.children:
            find_vp_in_sbar(tt)

    def find_np_in_sbar(t):
        global f
        global ff
        if t.id == 'VP':
            ff = False
        if t.id == 'NP' and f == True and ff == True:
            global np_sbar
            np_sbar = t
            f = False
        for tt in t.children:
            find_np_in_sbar(tt)

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

    parse_trees = parser.raw_parse(sent)
    global sent_list
    sent_list = [s for s in sent.split()]
    tree = next(parse_trees)[0]

    t = AnyNode(id='ROOT')
    make_tree(tree, t, sent_list)
    global sbar
    sbar = t
    global vp_sbar
    global f
    global ff
    global np_sbar
    global vp
    global np
    global vbz
    vp_sbar = []
    vp = t
    np = t
    vbz = 'bn2'
    np_sbar = t
    find_sbar(t)
    find_vp_in_sbar(sbar)
    f = True
    ff = True
    find_np_in_sbar(sbar)
    f = True
    find_vp(t)
    f = True
    find_np(t)
    f = True
    find_vbz(t)
    global simple_sentences
    simple_sentences = []
    simple_sentences.append([])
    make_sent(np)
    make_sent(vp)
    for i in range(len(vp_sbar)):
        simple_sentences.append([])
        if np_sbar == t:
            make_sent(np)
        else:
            make_sent(np_sbar)
        if vbz != 'bn2':
            simple_sentences[-1].append(vbz)
        make_sent(vp_sbar[i])

    simple = []
    for sentence in simple_sentences:
        string = ''
        for word in sentence:
            string += word + ' '
        string += '.'
        simple.append(string)

    def is_any_sbar(t):
        if t.id == 'SBAR':
            global f
            f = True
            return
        for tt in t.children:
            is_any_sbar(tt)

    f = False
    is_any_sbar(t)
    if f == False:
        simple = [sent]
    return simple