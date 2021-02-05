package com.nlp.tarpan.www.config;

import edu.stanford.nlp.pipeline.StanfordCoreNLP;
import org.springframework.context.annotation.Configuration;

import java.util.Properties;

/**
 * 中文词典配置
 *
 * @author sekift
 */
public class ChineseProperties {

    private static StanfordCoreNLP core = null;

    public static StanfordCoreNLP getCore() {
        return core;
    }

    static {
        init();
    }

    public static void init() {
        //构造一个StanfordCoreNLP对象，配置NLP的功能，如lemma是词干化，ner是命名实体识别等
        Properties props = new Properties();
        props.setProperty("annotators", "tokenize, ssplit, pos, lemma, ner, parse, coref");

        props.setProperty("tokenize.language", "zh");
        props.setProperty("segment.model", "edu/stanford/nlp/models/segmenter/chinese/ctb.gz");
        props.setProperty("segment.sighanCorporaDict", "edu/stanford/nlp/models/segmenter/chinese");
        props.setProperty("segment.serDictionary", "edu/stanford/nlp/models/segmenter/chinese/dict-chris6.ser.gz");
        props.setProperty("segment.sighanPostProcessing", "true");
        props.setProperty("pos.model", "edu/stanford/nlp/models/pos-tagger/chinese-distsim.tagger");
        props.setProperty("ner.language", "chinese");
        props.setProperty("ner.model", "edu/stanford/nlp/models/ner/chinese.misc.distsim.crf.ser.gz");
        props.setProperty("ner.applyNumericClassifiers", "true");
        props.setProperty("ner.useSUTime", "false");
        props.setProperty("ner.fine.regexner.mapping", "edu/stanford/nlp/models/kbp/chinese/gazetteers/cn_regexner_mapping.tab");
        props.setProperty("ner.fine.regexner.noDefaultOverwriteLabels", "CITY,COUNTRY,STATE_OR_PROVINCE");
        props.setProperty("parse.model", "edu/stanford/nlp/models/srparser/chineseSR.ser.gz");
        props.setProperty("depparse.model", "edu/stanford/nlp/models/parser/nndep/UD_Chinese.gz");
        props.setProperty("depparse.language", "chinese");
        props.setProperty("coref.sieves", "ChineseHeadMatch, ExactStringMatch, PreciseConstructs, StrictHeadMatch1, StrictHeadMatch2, StrictHeadMatch3, StrictHeadMatch4, PronounMatch");
        props.setProperty("coref.input.type", "raw");
        props.setProperty("coref.postprocessing", "true");
        props.setProperty("coref.calculateFeatureImportance", "false");
        props.setProperty("coref.useConstituencyTree", "true");
        props.setProperty("coref.useSemantics", "false");
        props.setProperty("coref.algorithm", "hybrid");
        props.setProperty("coref.path.word2vec", "");
        props.setProperty("coref.language", "zh");
        props.setProperty("coref.defaultPronounAgreement", "true");
        props.setProperty("coref.zh.dict", "edu/stanford/nlp/models/dcoref/zh-attributes.txt.gz");
        props.setProperty("coref.print.md.log", "false");
        props.setProperty("coref.md.type", "RULE");
        props.setProperty("coref.md.liberalChineseMD", "false");
        props.setProperty("kbp.semgrex", "edu/stanford/nlp/models/kbp/chinese/semgrex");
        props.setProperty("kbp.tokensregex", "edu/stanford/nlp/models/kbp/chinese/tokensregex");
        props.setProperty("kbp.language", "zh");
        props.setProperty("kbp.model", "none");
        props.setProperty("entitylink.wikidict", "edu/stanford/nlp/models/kbp/chinese/wikidict_chinese.tsv.gz");
        core = new StanfordCoreNLP(props);
    }
}
