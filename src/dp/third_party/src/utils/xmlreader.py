from xml.dom import minidom
import os

'''
Read xml output from Stanford CoreNLP and extract
1. Sentence index
2. Token index (within sentence)
3. Token
4. Lemma
5. POS
6. Dependency label
7. Dependency head
8. NER
9. Partial constituent parse
'''


class TokenElem(object):
    """ Data structure for each token
    """

    def __init__(self, idx, word, lemma, pos, begin_offset, end_offset, nertype=None):
        self.word, self.pos = word, pos
        self.idx, self.lemma = idx, lemma
        self.deptype, self.headidx = None, None
        self.nertype = nertype
        self.begin_offset = begin_offset
        self.end_offset = end_offset
        self.partialparse = None


class SentElem(object):
    """ Data structure for each sentence
    """

    def __init__(self, idx, tokenlist):
        self.tokenlist = tokenlist
        self.idx = idx


class DepElem(object):
    """ Data structure for reading dependency parsing
    """

    def __init__(self, deptype, gidx, gtoken, didx, dtoken):
        self.deptype = deptype
        self.gidx, self.gtoken = gidx, gtoken
        self.didx, self.dtoken = didx, dtoken


def getText(nodelist):
    rc = []
    for node in nodelist:
        if node.nodeType == node.TEXT_NODE:
            rc.append(node.data)
    return ''.join(rc)


def getTokens(sent):
    tokens = sent.getElementsByTagName('token')
    tokenelem_dict = {}
    for token in tokens:
        pos = getText(token.getElementsByTagName('POS')[0].childNodes)
        word = getText(token.getElementsByTagName('word')[0].childNodes)
        lemma = getText(token.getElementsByTagName('lemma')[0].childNodes)
        begin_offset = int(getText(token.getElementsByTagName('CharacterOffsetBegin')[0].childNodes))
        end_offset = int(getText(token.getElementsByTagName('CharacterOffsetEnd')[0].childNodes))
        try:
            ner = getText(token.getElementsByTagName('NER')[0].childNodes)
        except IndexError:
            ner = None
        idx = int(token.attributes['id'].value)
        token = TokenElem(idx, word, lemma, pos, begin_offset, end_offset, ner)
        tokenelem_dict[idx] = token
    return tokenelem_dict


def getConstituenttree(sent):
    tree = sent.getElementsByTagName('parse')
    tree = getText(tree[0].childNodes)
    return tree


def getDeptree(sent):
    deps_group = sent.getElementsByTagName('dependencies')
    for item in deps_group:
        if item.attributes['type'].value == 'basic-dependencies':
            deps = item.getElementsByTagName('dep')
        else:
            # print item.attributes['type'].value
            pass
    depelem_list = []
    for dep in deps:
        deptype = dep.attributes['type'].value
        governor = dep.getElementsByTagName('governor')
        gidx = int(governor[0].attributes['idx'].value)
        gtoken = getText(governor[0].childNodes)
        dependent = dep.getElementsByTagName('dependent')
        didx = int(dependent[0].attributes['idx'].value)
        dtoken = getText(dependent[0].childNodes)
        elem = DepElem(deptype, gidx, gtoken, didx, dtoken)
        depelem_list.append(elem)
    return depelem_list


def integrate(token_dict, dep_list):
    """ Integrate dependency information into token list
    """
    for dep in dep_list:
        deptype = dep.deptype
        gidx, gtoken = dep.gidx, dep.gtoken
        didx, dtoken = dep.didx, dep.dtoken
        tokenelem = token_dict[didx]
        tokenelem.deptype = deptype
        tokenelem.headidx = gidx
        token_dict[didx] = tokenelem
    token_list = []
    for idx in range(len(token_dict)):
        token_list.append(token_dict[idx + 1])
    return token_list


def reader(fname):
    try:
        xmldoc = minidom.parse(fname)
        sentelem_list = []
        constituent_list = []
        sentlist = xmldoc.getElementsByTagName('sentences')[0].getElementsByTagName('sentence')
        for (idx, sent) in enumerate(sentlist):
            tokenelem_dict = getTokens(sent)
            tree = getConstituenttree(sent)
            constituent_list.append(tree)
            depelem_list = getDeptree(sent)
            tokenelem_list = integrate(tokenelem_dict, depelem_list)
            sentelem_list.append(SentElem(idx, tokenelem_list))
        return sentelem_list, constituent_list
    except:
        print('Could not parse:', fname)
        return None, None
        #os.remove(fname)
        #os.remove(fname.replace('.out.text.xml', '.out'))
        #os.remove(fname.replace('.out.text.xml', '.out.dis'))
        #os.remove(fname.replace('.out.text.xml', '.out.edus'))
        #os.remove(fname.replace('.out.text.xml', '.out.text'))


def combineparse2sent(sent, parse):
    """ Combine constitent parse into sent
    """
    parse = parse.split()
    tokenlist = [token.word for token in sent.tokenlist]
    parselist, tidx = [""] * len(tokenlist), 0
    while parse:
        item = parse.pop(0)
        parselist[tidx] += (" " + item)
        partialparse = parselist[tidx].replace(' ', '')
        word = tokenlist[tidx].replace(' ', '')
        # print word, partialparse
        if (word + ')') in partialparse:
            tidx += 1
    # Attach to sent
    for (tidx, token) in enumerate(sent.tokenlist):
        item = parselist[tidx]
        sent.tokenlist[tidx].partialparse = item
    return sent


def combine(sent_list, const_list):
    """
    """
    for (sidx, sent) in enumerate(sent_list):
        parse = const_list[sidx]
        sent = combineparse2sent(sent, parse)
        sent_list[sidx] = sent
    return sent_list


def writer(sent_list, fconll):
    with open(fconll, 'w') as fout:
        for sent in sent_list:
            for token in sent.tokenlist:
                line = str(sent.idx) + '\t' + str(token.idx) + '\t' + token.word + '\t' + token.lemma \
                       + '\t' + str(token.pos) + '\t' + str(token.deptype) + '\t' + str(token.headidx) \
                       + '\t' + str(token.nertype) + '\t' + str(token.partialparse) + '\t' \
                       + str(token.begin_offset) + '\t' + str(token.end_offset) + '\n'
                fout.write(line)
            fout.write('\n')


if __name__ == '__main__':
    sent_list, const_list = reader('test.xml')
    sent_list = combine(sent_list, const_list)
    writer(sent_list, 'test.conll')
