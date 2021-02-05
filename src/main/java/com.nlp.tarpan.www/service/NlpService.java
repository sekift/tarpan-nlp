package com.nlp.tarpan.www.service;

import java.util.Map;

public interface NlpService {
    /**
     * 分词、词性与依存关系
     * @param input
     * @return
     */
    public Map<String, String> parser(String input);
}
