package com.nlp.tarpan.www.service.impl;

import com.nlp.tarpan.www.config.ChineseProperties;
import com.nlp.tarpan.www.service.NlpService;
import edu.stanford.nlp.ling.CoreAnnotations;
import edu.stanford.nlp.ling.CoreLabel;
import edu.stanford.nlp.pipeline.Annotation;
import edu.stanford.nlp.pipeline.StanfordCoreNLP;
import edu.stanford.nlp.semgraph.SemanticGraph;
import edu.stanford.nlp.semgraph.SemanticGraphCoreAnnotations;
import edu.stanford.nlp.util.CoreMap;
import org.springframework.stereotype.Service;

import java.util.HashMap;
import java.util.List;
import java.util.Map;

@Service
public class NlpServiceImpl implements NlpService {

    private static StanfordCoreNLP pipeline = ChineseProperties.getCore();

    @Override
    public Map<String, String> parser(String input) {
        Map<String, String> resultMap = new HashMap<>(4);
        StringBuilder wordBuilder = new StringBuilder();
        StringBuilder posBuilder = new StringBuilder();
        StringBuilder depBuilder = new StringBuilder();

        // 创造一个空的Annotation对象
        Annotation document = new Annotation(input);
        // 对文本进行分析
        pipeline.annotate(document);
        //获取文本处理结果
        List<CoreMap> sentences = document.get(CoreAnnotations.SentencesAnnotation.class);
        for (CoreMap sentence : sentences) {
            for (CoreLabel token : sentence.get(CoreAnnotations.TokensAnnotation.class)) {
                // 获取句子的token（可以是作为分词后的词语）
                String word = token.get(CoreAnnotations.TextAnnotation.class);
                wordBuilder.append(word).append(" ");
                //词性标注
                String pos = token.get(CoreAnnotations.PartOfSpeechAnnotation.class);
                posBuilder.append(word + "#" + pos).append(" ");
            }
            // 句子的依赖图
            SemanticGraph graph = sentence.get(SemanticGraphCoreAnnotations.CollapsedCCProcessedDependenciesAnnotation.class);
            String parsed = graph.toString(SemanticGraph.OutputFormat.LIST).replace("\n", "   ");
            depBuilder.append(parsed);
        }
        resultMap.put("seged", wordBuilder.toString().trim());
        resultMap.put("posed", posBuilder.toString().trim());
        resultMap.put("parsed", depBuilder.toString().trim());
        return resultMap;
    }
}
