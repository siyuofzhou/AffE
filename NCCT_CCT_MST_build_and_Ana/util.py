# – coding: gbk –

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
    :param graph: 输入图 graph
    :return:  返回每一对r1,r2的头尾节点列表，用字典保存
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