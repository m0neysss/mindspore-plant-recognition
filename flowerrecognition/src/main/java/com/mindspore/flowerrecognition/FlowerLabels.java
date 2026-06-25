package com.mindspore.flowerrecognition;

import java.util.Arrays;
import java.util.Collections;
import java.util.List;

public final class FlowerLabels {
    public static final List<FlowerLabel> ALL = Collections.unmodifiableList(Arrays.asList(
            new FlowerLabel("astilbe", "落新妇"),
            new FlowerLabel("bellflower", "风铃草"),
            new FlowerLabel("black_eyed_susan", "黑心金光菊"),
            new FlowerLabel("calendula", "金盏花"),
            new FlowerLabel("california_poppy", "花菱草"),
            new FlowerLabel("carnation", "康乃馨"),
            new FlowerLabel("common_daisy", "雏菊"),
            new FlowerLabel("coreopsis", "金鸡菊"),
            new FlowerLabel("daffodil", "水仙"),
            new FlowerLabel("dandelion", "蒲公英"),
            new FlowerLabel("iris", "鸢尾"),
            new FlowerLabel("magnolia", "木兰"),
            new FlowerLabel("rose", "玫瑰"),
            new FlowerLabel("sunflower", "向日葵"),
            new FlowerLabel("tulip", "郁金香"),
            new FlowerLabel("water_lily", "睡莲")
    ));

    private FlowerLabels() {
    }

    public static final class FlowerLabel {
        private final String englishName;
        private final String chineseName;

        private FlowerLabel(String englishName, String chineseName) {
            this.englishName = englishName;
            this.chineseName = chineseName;
        }

        public String getEnglishName() {
            return englishName;
        }

        public String getChineseName() {
            return chineseName;
        }
    }
}
