package com.nlp.tarpan.www.test;

import edu.stanford.nlp.semgraph.SemanticGraph;

public class TestGraph {
    private static final String STR = "-> 差/VV (root)\n" +
            "  -> 酒店/NN (nsubj)\n" +
            "  -> 实在/AD (advmod)\n" +
            "  -> 房间/NN (dobj)\n" +
            "  -> 小/VA (conj)\n" +
            "    -> 又/AD (advmod)\n" +
            "    -> 脏/VA (conj)\n" +
            "      -> 又/AD (advmod)\n" +
            "      -> 环境/NN (dobj)\n" +
            "        -> 卫生间/NN (compound:nn)\n" +
            "  -> 差/VV (conj)\n" +
            "    -> 太/AD (advmod)\n" +
            "    -> 酒店/NN (dobj)\n" +
            "      -> 整/DT (det)\n" +
            "        -> 个/M (mark:clf)\n" +
            "    -> 像/VV (conj)\n" +
            "      -> 有点/AD (advmod)\n" +
            "      -> 招待所/NN (dobj)\n" +
            "        -> 马路/NN (nmod)\n" +
            "          -> 边上/LC (case)\n" +
            "          -> 的/DEG (case)";


    public static void main(String[] args) {
        SemanticGraph graph = new SemanticGraph();
        System.out.println(graph.toList());

    }
}
