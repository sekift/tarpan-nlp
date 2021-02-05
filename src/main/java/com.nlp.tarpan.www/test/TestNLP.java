package com.nlp.tarpan.www.test;

import com.nlp.tarpan.www.config.ChineseProperties;
import edu.stanford.nlp.ling.CoreAnnotations.PartOfSpeechAnnotation;
import edu.stanford.nlp.ling.CoreAnnotations.SentencesAnnotation;
import edu.stanford.nlp.ling.CoreAnnotations.TextAnnotation;
import edu.stanford.nlp.ling.CoreAnnotations.TokensAnnotation;
import edu.stanford.nlp.ling.CoreLabel;
import edu.stanford.nlp.pipeline.Annotation;
import edu.stanford.nlp.pipeline.StanfordCoreNLP;
import edu.stanford.nlp.semgraph.SemanticGraph;
import edu.stanford.nlp.semgraph.SemanticGraphCoreAnnotations;
import edu.stanford.nlp.util.CoreMap;

import java.util.HashMap;
import java.util.List;
import java.util.Map;

public class TestNLP {

    public static Map<String, String> test() {
        StanfordCoreNLP pipeline = ChineseProperties.getCore();
        // 待处理字符串  
        String text = "酒店实在差 房间又小又脏 卫生间环境太差 整个酒店有点像马路边上的招待所";
        // 创造一个空的Annotation对象  
        Annotation document = new Annotation(text);
        // 对文本进行分析
        pipeline.annotate(document);
        //获取文本处理结果  
        List<CoreMap> sentences = document.get(SentencesAnnotation.class);
        Map<String, String> resultMap = new HashMap<>(4);
        StringBuilder wordBuilder = new StringBuilder();
        StringBuilder posBuilder = new StringBuilder();
        StringBuilder depBuilder = new StringBuilder();
        for (CoreMap sentence : sentences) {

            for (CoreLabel token : sentence.get(TokensAnnotation.class)) {
                // 获取句子的token（可以是作为分词后的词语）  
                String word = token.get(TextAnnotation.class);
                wordBuilder.append(word).append(" ");
                //词性标注  
                String pos = token.get(PartOfSpeechAnnotation.class);
                posBuilder.append(word+"#"+pos).append(" ");
            }
            // 句子的依赖图
            SemanticGraph graph = sentence.get(SemanticGraphCoreAnnotations.CollapsedCCProcessedDependenciesAnnotation.class);
            String parsed = graph.toString(SemanticGraph.OutputFormat.LIST).replace("\n","   ");
            depBuilder.append(parsed);
        }
        resultMap.put("seged", wordBuilder.toString().trim());
        resultMap.put("posed", posBuilder.toString().trim());
        resultMap.put("parsed",depBuilder.toString().trim());
        return resultMap;
    }

    public static void main(String[] args) {
        System.out.println(TestNLP.test());
    }

}  