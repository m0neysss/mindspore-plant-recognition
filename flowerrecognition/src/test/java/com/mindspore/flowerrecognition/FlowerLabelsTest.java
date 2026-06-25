package com.mindspore.flowerrecognition;

import org.junit.Test;

import java.util.Arrays;
import java.util.List;

import static org.junit.Assert.assertEquals;

public class FlowerLabelsTest {
    @Test
    public void labelsKeepModelOutputOrder() {
        List<String> expectedEnglish = Arrays.asList(
                "astilbe", "bellflower", "black_eyed_susan", "calendula",
                "california_poppy", "carnation", "common_daisy", "coreopsis",
                "daffodil", "dandelion", "iris", "magnolia", "rose", "sunflower",
                "tulip", "water_lily");
        List<String> expectedChinese = Arrays.asList(
                "落新妇", "风铃草", "黑心金光菊", "金盏花", "花菱草", "康乃馨", "雏菊",
                "金鸡菊", "水仙", "蒲公英", "鸢尾", "木兰", "玫瑰", "向日葵", "郁金香", "睡莲");

        assertEquals(16, FlowerLabels.ALL.size());
        for (int index = 0; index < FlowerLabels.ALL.size(); index++) {
            assertEquals(expectedEnglish.get(index), FlowerLabels.ALL.get(index).getEnglishName());
            assertEquals(expectedChinese.get(index), FlowerLabels.ALL.get(index).getChineseName());
        }
    }

    @Test(expected = UnsupportedOperationException.class)
    public void labelsCannotBeReordered() {
        FlowerLabels.ALL.clear();
    }
}
