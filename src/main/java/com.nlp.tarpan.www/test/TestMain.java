package com.nlp.tarpan.www.test;

import org.jsoup.Jsoup;
import org.jsoup.nodes.Document;

import java.net.URL;

public class TestMain {
    public static void main(String[] args) {
        try {
            Document document = Jsoup.parse(new URL("https://github.com/hankcs/HanLP/tree/1.x/src/test/java/com/hankcs/demo"), 30*1000);
            System.out.println(document);
        }catch(Exception e){
            e.printStackTrace();
        }
    }
}
