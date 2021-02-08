package com.nlp.tarpan.www.config;

import org.springframework.boot.ApplicationArguments;
import org.springframework.boot.ApplicationRunner;
import org.springframework.stereotype.Component;

/**
 * @description: 一些初始化方法
 */
@Component
public class ApplicationRunnerImpl implements ApplicationRunner {

    @Override
    public void run(ApplicationArguments args) throws Exception {
        ChineseProperties.getCore();
        System.out.println("Stanford分词程序启动完毕");
    }
}