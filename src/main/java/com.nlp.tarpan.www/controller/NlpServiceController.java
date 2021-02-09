package com.nlp.tarpan.www.controller;

import com.nlp.tarpan.www.service.NlpService;
import org.springframework.beans.factory.annotation.Autowired;
import org.springframework.web.bind.annotation.PostMapping;
import org.springframework.web.bind.annotation.RequestMapping;
import org.springframework.web.bind.annotation.RestController;

import java.util.List;
import java.util.Map;

@RestController
@RequestMapping("/nlp")
public class NlpServiceController {

    @Autowired
    private NlpService nlpService;

    @PostMapping("/parser")
    public Map<String, List<String>> parser(String input) {
        return nlpService.parser(input);
    }
}
