import maxflow




if __name__ == '__main__':
    print 'python maxflow demo...'
    ##print maxflow.__version__
    g=maxflow.Graph[float](1,2)
    nodes=g.add_nodes(2)
    g.add_edge(nodes[0],nodes[1],1,2)
    
    g.add_tedge(nodes[0],2,5)
    g.add_tedge(nodes[1],9,4)
    
    flow = g.maxflow()
    print "Maximum flow:", flow

    print "Segment of the node 0:", g.get_segment(nodes[0])
    print "Segment of the node 1:", g.get_segment(nodes[1])