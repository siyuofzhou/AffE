# �C coding: gbk �C

def add(k, v, dic):
    if k in dic:
        dic[k].append(v)
    else:
        dic[k] = [v]

def addSet(k, v, dic):
    if k in dic:
        dic[k].add(v)
    else:
        dic[k] = {v}

def find_head_and_tail(graph):
    '''
    :param graph: ����ͼ graph
    :return:  ����ÿһ��r1,r2��ͷβ�ڵ��б����ֵ䱣��
    '''
    hrrt = {}
    nodes = [-1] * 3
    rels = [-1] * 2

    def dfs(graph, now, ni, ri):
        if ri == 2:
            if now != nodes[0]:
                add((nodes[0], rels[0], rels[1]), now, hrrt)
            return
        nodes[ni] = now
        if now in graph:
            for r, t in graph[now]:
                rels[ri] = r
                dfs(graph, t, ni + 1, ri + 1)

    for e in graph.keys():
        dfs(graph, e, 0, 0)
    return hrrt