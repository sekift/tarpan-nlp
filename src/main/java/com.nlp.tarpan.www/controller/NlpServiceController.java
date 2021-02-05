package com.nlp.tarpan.www.controller;

import com.nlp.tarpan.www.service.NlpService;
import org.springframework.beans.factory.annotation.Autowired;
import org.springframework.web.bind.annotation.GetMapping;
import org.springframework.web.bind.annotation.RequestMapping;
import org.springframework.web.bind.annotation.RestController;

import java.util.Map;

@RestController
@RequestMapping("/nlp")
public class NlpServiceController {

    @Autowired
    private NlpService nlpService;

    @GetMapping("/parser")
    public Map<String, String> parser(String input){
        return nlpService.parser(input);
    }
}
