package com.nlp.tarpan.www.service;

import java.util.List;
import java.util.Map;

public interface NlpService {
    /**
     * 分词、词性与依存关系
     *
     * @param input
     * @return
     */
    public Map<String, List<String>> parser(String input);
}
