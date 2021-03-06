package com.nlp.tarpan.www.test;

import edu.stanford.nlp.coref.CorefCoreAnnotations;
import edu.stanford.nlp.coref.data.CorefChain;
import edu.stanford.nlp.io.IOUtils;
import edu.stanford.nlp.ling.CoreAnnotations;
import edu.stanford.nlp.ling.CoreLabel;
import edu.stanford.nlp.ling.IndexedWord;
import edu.stanford.nlp.pipeline.Annotation;
import edu.stanford.nlp.pipeline.StanfordCoreNLP;
import edu.stanford.nlp.semgraph.SemanticGraph;
import edu.stanford.nlp.semgraph.SemanticGraphCoreAnnotations;
import edu.stanford.nlp.semgraph.SemanticGraphEdge;
import edu.stanford.nlp.sentiment.SentimentCoreAnnotations;
import edu.stanford.nlp.trees.Tree;
import edu.stanford.nlp.trees.TreeCoreAnnotations;
import edu.stanford.nlp.util.CoreMap;

import java.io.IOException;
import java.io.PrintWriter;
import java.util.List;
import java.util.Map;
import java.util.Properties;

/** This class demonstrates building and using a Stanford CoreNLP pipeline. */
public class StanfordCoreNlpDemo {

  private StanfordCoreNlpDemo() { } // static meain metho

  /** Usage: java -cp "*" StanfordCoreNlpDemo [inputFile [outputTextFile [outputXmlFile]]] */
  public static void main(String[] args) throws IOException {
    // set up optional output files
    PrintWriter out;
    if (args.length > 1) {
      out = new PrintWriter(args[1]);
    } else {
      out = new PrintWriter(System.out);
    }
    PrintWriter xmlOut = null;
    if (args.length > 2) {
      xmlOut = new PrintWriter(args[2]);
    }

    // Create a CoreNLP pipeline. To build the default pipeline, you can just use:
    //   StanfordCoreNLP pipeline = new StanfordCoreNLP(props);
    // Here's a more complex setup example:
    //   Properties props = new Properties();
    //   props.put("annotators", "tokenize, ssplit, pos, lemma, ner, depparse");
    //   props.put("ner.model", "edu/stanford/nlp/models/ner/english.all.3class.distsim.crf.ser.gz");
    //   props.put("ner.applyNumericClassifiers", "false");
    //   StanfordCoreNLP pipeline = new StanfordCoreNLP(props);

    // Add in sentiment
    Properties props = new Properties();
    props.setProperty("annotators", "tokenize, ssplit, pos, lemma, ner, parse, coref");
    props.setProperty("tokenize.language","zh");
    props.setProperty("segment.model","edu/stanford/nlp/models/segmenter/chinese/ctb.gz");
    props.setProperty("segment.sighanCorporaDict","edu/stanford/nlp/models/segmenter/chinese");
    props.setProperty("segment.serDictionary","edu/stanford/nlp/models/segmenter/chinese/dict-chris6.ser.gz");
    props.setProperty("segment.sighanPostProcessing","true");
    props.setProperty("pos.model","edu/stanford/nlp/models/pos-tagger/chinese-distsim.tagger");
    props.setProperty("ner.language","chinese");
    props.setProperty("ner.model","edu/stanford/nlp/models/ner/chinese.misc.distsim.crf.ser.gz");
    props.setProperty("ner.applyNumericClassifiers","true");
    props.setProperty("ner.useSUTime","false");
    props.setProperty("ner.fine.regexner.mapping","edu/stanford/nlp/models/kbp/chinese/gazetteers/cn_regexner_mapping.tab");
    props.setProperty("ner.fine.regexner.noDefaultOverwriteLabels","CITY,COUNTRY,STATE_OR_PROVINCE");
    props.setProperty("parse.model","edu/stanford/nlp/models/srparser/chineseSR.ser.gz");
    props.setProperty("depparse.model","edu/stanford/nlp/models/parser/nndep/UD_Chinese.gz");
    props.setProperty("depparse.language","chinese");
    props.setProperty("coref.sieves","ChineseHeadMatch, ExactStringMatch, PreciseConstructs, StrictHeadMatch1, StrictHeadMatch2, StrictHeadMatch3, StrictHeadMatch4, PronounMatch");
    props.setProperty("coref.input.type","raw");
    props.setProperty("coref.postprocessing","true");
    props.setProperty("coref.calculateFeatureImportance","false");
    props.setProperty("coref.useConstituencyTree","true");
    props.setProperty("coref.useSemantics","false");
    props.setProperty("coref.algorithm","hybrid");
    props.setProperty("coref.path.word2vec","");
    props.setProperty("coref.language","zh");
    props.setProperty("coref.defaultPronounAgreement","true");
    props.setProperty("coref.zh.dict","edu/stanford/nlp/models/dcoref/zh-attributes.txt.gz");
    props.setProperty("coref.print.md.log","false");
    props.setProperty("coref.md.type","RULE");
    props.setProperty("coref.md.liberalChineseMD","false");
    props.setProperty("kbp.semgrex","edu/stanford/nlp/models/kbp/chinese/semgrex");
    props.setProperty("kbp.tokensregex","edu/stanford/nlp/models/kbp/chinese/tokensregex");
    props.setProperty("kbp.language","zh");
    props.setProperty("kbp.model","none");
    props.setProperty("entitylink.wikidict","edu/stanford/nlp/models/kbp/chinese/wikidict_chinese.tsv.gz");

    StanfordCoreNLP pipeline = new StanfordCoreNLP(props);

    // Initialize an Annotation with some text to be annotated. The text is the argument to the constructor.
    Annotation annotation;
    if (args.length > 0) {
      annotation = new Annotation(IOUtils.slurpFileNoExceptions(args[0]));
    } else {
      annotation = new Annotation("酒店的硬件设施比较差,不符合我们对环境的要求.细节就不说了.服务也很差,基本上就象个招待所.而且没有热水.");
    }

    // run all the selected Annotators on this text
    pipeline.annotate(annotation);

    // this prints out the results of sentence analysis to file(s) in good formats
    pipeline.prettyPrint(annotation, out);
    if (xmlOut != null) {
      pipeline.xmlPrint(annotation, xmlOut);
    }

    // Access the Annotation in code
    // The toString() method on an Annotation just prints the text of the Annotation
    // But you can see what is in it with other methods like toShorterString()
    out.println();
    out.println("The top level annotation");
    out.println(annotation.toShorterString());
    out.println();

    // An Annotation is a Map with Class keys for the linguistic analysis types.
    // You can get and use the various analyses individually.
    // For instance, this gets the parse tree of the first sentence in the text.
    List<CoreMap> sentences = annotation.get(CoreAnnotations.SentencesAnnotation.class);
    if (sentences != null && ! sentences.isEmpty()) {
      CoreMap sentence = sentences.get(0);
      out.println("The keys of the first sentence's CoreMap are:");
      out.println(sentence.keySet());
      out.println();
      out.println("The first sentence is:");
      out.println(sentence.toShorterString());
      out.println();
      out.println("The first sentence tokens are:");
      for (CoreMap token : sentence.get(CoreAnnotations.TokensAnnotation.class)) {
        out.println(token.toShorterString());
      }
      Tree tree = sentence.get(TreeCoreAnnotations.TreeAnnotation.class);
      out.println();
      out.println("The first sentence parse tree is:");
      tree.pennPrint(out);
      out.println();
      out.println("The first sentence basic dependencies are:");
      out.println(sentence.get(SemanticGraphCoreAnnotations.BasicDependenciesAnnotation.class).toString(SemanticGraph.OutputFormat.LIST));
      out.println("The first sentence collapsed, CC-processed dependencies are:");
      SemanticGraph graph = sentence.get(SemanticGraphCoreAnnotations.CollapsedCCProcessedDependenciesAnnotation.class);
      out.println(graph.toString(SemanticGraph.OutputFormat.LIST));

      // Print out dependency structure around one word
      // This give some idea of how to navigate the dependency structure in a SemanticGraph
      IndexedWord node = graph.getNodeByIndexSafe(5);
      // The below way also works
      // IndexedWord node = new IndexedWord(sentences.get(0).get(CoreAnnotations.TokensAnnotation.class).get(5 - 1));
      out.println("Printing dependencies around \"" + node.word() + "\" index " + node.index());
      List<SemanticGraphEdge> edgeList = graph.getIncomingEdgesSorted(node);
      if (! edgeList.isEmpty()) {
        assert edgeList.size() == 1;
        int head = edgeList.get(0).getGovernor().index();
        String headWord = edgeList.get(0).getGovernor().word();
        String deprel = edgeList.get(0).getRelation().toString();
        out.println("Parent is word \"" + headWord + "\" index " + head + " via " + deprel);
      } else  {
        out.println("Parent is ROOT via root");
      }
      edgeList = graph.outgoingEdgeList(node);
      for (SemanticGraphEdge edge : edgeList) {
        String depWord = edge.getDependent().word();
        int depIdx = edgeList.get(0).getDependent().index();
        String deprel = edge.getRelation().toString();
        out.println("Child is \"" + depWord + "\" (" + depIdx + ") via " + deprel);
      }
      out.println();


      // Access coreference. In the coreference link graph,
      // each chain stores a set of mentions that co-refer with each other,
      // along with a method for getting the most representative mention.
      // Both sentence and token offsets start at 1!
      out.println("Coreference information");
      Map<Integer, CorefChain> corefChains =
          annotation.get(CorefCoreAnnotations.CorefChainAnnotation.class);
      if (corefChains == null) { return; }
      for (Map.Entry<Integer,CorefChain> entry: corefChains.entrySet()) {
        out.println("Chain " + entry.getKey());
        for (CorefChain.CorefMention m : entry.getValue().getMentionsInTextualOrder()) {
          // We need to subtract one since the indices count from 1 but the Lists start from 0
          List<CoreLabel> tokens = sentences.get(m.sentNum - 1).get(CoreAnnotations.TokensAnnotation.class);
          // We subtract two for end: one for 0-based indexing, and one because we want last token of mention not one following.
          out.println("  " + m + ", i.e., 0-based character offsets [" + tokens.get(m.startIndex - 1).beginPosition() +
                  ", " + tokens.get(m.endIndex - 2).endPosition() + ')');
        }
      }
      out.println();

      out.println("The first sentence overall sentiment rating is " + sentence.get(SentimentCoreAnnotations.SentimentClass.class));
    }
    IOUtils.closeIgnoringExceptions(out);
    IOUtils.closeIgnoringExceptions(xmlOut);
  }

}
