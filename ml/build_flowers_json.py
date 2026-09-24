"""构建花卉百科数据 backend/app/data/flowers.json（102 类中文资料）。

对应 FR-02 与 docs/03 §5。
数据组织方式
------------
1. **逐种事实**（``SPECIES``）：中文名、科、属、花色、花期、简介 —— 逐条核对，按英文名索引。
   以英文名为键而非 class_id，这样即使将来数据源标签顺序变化，映射仍由构建期断言兜住。
2. **按类养护模板**（``CARE_BY_CATEGORY``）：光照、浇水、土壤、繁殖、养护要点按植物**类别**
   （球根 / 水生 / 兰科 / 灌木 / 藤本 / 一二年生 / 宿根 / 野生草花 / 多肉附生）给通用建议。
   这样做是刻意为之：把"可核实的物种事实"与"按类别成立的通用养护"分开，
   避免为 102 种花编造不可靠的逐种养护细节。

用法
----
    python -m ml.build_flowers_json                     # 只生成 flowers.json
    python -m ml.build_flowers_json --export-thumbs     # 同时导出 102 张百科缩略图
    python -m ml.build_flowers_json --strict            # 缺字段即报错退出（当前应通过）
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from ml.dataset import get_class_names, load_manifest
from ml.utils import save_json

OUT_JSON = Path("backend/app/data/flowers.json")
THUMB_DIR = Path("backend/app/static/flowers")
THUMB_SIZE = 320


# --------------------------------------------------------------------------
# 按类别的通用养护模板
# --------------------------------------------------------------------------
CARE_BY_CATEGORY: dict[str, dict[str, str]] = {
    "orchid": {
        "category_cn": "兰科附生植物",
        "light": "明亮散射光，忌强光直射",
        "watering": "基质见干见湿，忌长期积水",
        "soil": "树皮、水苔、陶粒等透气基质",
        "propagation": "分株为主，也可组织培养",
        "care_tips": "关键是通风与透气：忌闷热、忌盆内积水，花后及时剪除残花梗。",
    },
    "bulb": {
        "category_cn": "球根 / 块茎类",
        "light": "全日照至半阴",
        "watering": "生长期保持微润，休眠期严格控水",
        "soil": "疏松、排水良好的沙质壤土",
        "propagation": "分球、分株",
        "care_tips": "花后务必保留叶片继续养球，待叶片自然枯黄后起球或断水；休眠期积水最易烂球。",
    },
    "aquatic": {
        "category_cn": "水生植物",
        "light": "全日照，每天至少 6 小时直射光",
        "watering": "始终保持浅水层或基质饱和湿润",
        "soil": "塘泥或水生植物专用基质",
        "propagation": "分株、播种、根茎切割",
        "care_tips": "光照不足会只长叶不开花；冬季需防冻，水体不可完全干涸。",
    },
    "succulent": {
        "category_cn": "多肉 / 附生植物",
        "light": "充足光照，盛夏午后适度遮阴",
        "watering": "干透浇透，宁干勿湿",
        "soil": "颗粒土为主，排水极佳",
        "propagation": "扦插、分株、侧芽分离",
        "care_tips": "最怕长期潮湿与低温高湿；冬季控水并保持 10 ℃ 以上更安全。",
    },
    "shrub": {
        "category_cn": "灌木 / 小乔木",
        "light": "全日照至半阴",
        "watering": "生长期保持土壤湿润，忌积水",
        "soil": "肥沃疏松、微酸性至中性壤土",
        "propagation": "扦插、压条、嫁接",
        "care_tips": "花后适度修剪促发新枝；注意通风，高温高湿期易发叶斑病与蚧壳虫。",
    },
    "vine": {
        "category_cn": "藤本植物",
        "light": "全日照至半阴",
        "watering": "生长期充足浇水，保持湿润",
        "soil": "肥沃、排水良好的壤土",
        "propagation": "扦插、压条",
        "care_tips": "需设支架或牵引绳；定期修剪控制长势，否则易只长藤不开花。",
    },
    "annual": {
        "category_cn": "一二年生草花",
        "light": "全日照",
        "watering": "保持土壤湿润，忌忽干忽湿",
        "soil": "肥沃疏松的壤土",
        "propagation": "播种",
        "care_tips": "及时摘除残花可显著延长花期；高秆品种注意防倒伏。",
    },
    "perennial": {
        "category_cn": "多年生宿根草本",
        "light": "全日照至半阴",
        "watering": "生长期保持湿润，冬季减少浇水",
        "soil": "肥沃、排水良好的壤土",
        "propagation": "分株、扦插、播种",
        "care_tips": "花后修剪残花与老枝；越冬前清理地上枯枝，减少病虫越冬基数。",
    },
    "wildflower": {
        "category_cn": "野生 / 宿根草花",
        "light": "全日照",
        "watering": "耐旱，见干见湿即可",
        "soil": "排水良好的普通园土，不择土壤",
        "propagation": "播种、分株、自播繁殖",
        "care_tips": "管理粗放，忌积水与过量施肥，否则易徒长倒伏。",
    },
}


# --------------------------------------------------------------------------
# 102 种花卉事实（键 = 英文名，与 ml/dataset.py::CLASS_NAMES_EN 严格对应）
# --------------------------------------------------------------------------
SPECIES: dict[str, dict[str, str]] = {
    "pink primrose": {"cn": "粉报春", "family": "报春花科", "genus": "报春花属", "color": "粉紫", "season": "早春", "cat": "perennial",
                      "desc": "报春花属多年生草本，花冠粉紫色、喉部带黄心，是早春最早开放的花卉之一，常成丛生于林缘湿地。"},
    "hard-leaved pocket orchid": {"cn": "硬叶兜兰", "family": "兰科", "genus": "兜兰属", "color": "白绿带紫斑", "season": "春夏季", "cat": "orchid",
                      "desc": "地生或半附生兰，叶片革质硬挺并带斑纹，唇瓣膨大呈囊状（兜状），是兜兰属的典型特征。"},
    "canterbury bells": {"cn": "风铃草", "family": "桔梗科", "genus": "风铃草属", "color": "蓝紫、白", "season": "夏季", "cat": "perennial",
                      "desc": "花冠钟形下垂，聚生成塔状花序，蓝紫色最为常见，是欧洲传统庭园花境植物。"},
    "sweet pea": {"cn": "香豌豆", "family": "豆科", "genus": "山黧豆属", "color": "粉、紫、白、红", "season": "春夏季", "cat": "vine",
                      "desc": "一年生攀缘草本，蝶形花具浓郁甜香，是重要的切花与芳香花卉。"},
    "english marigold": {"cn": "金盏花", "family": "菊科", "genus": "金盏花属", "color": "橙黄", "season": "春至秋", "cat": "annual",
                      "desc": "头状花序橙黄或金黄色，花瓣可食用并含类胡萝卜素，欧洲传统药用与观赏花卉。"},
    "tiger lily": {"cn": "卷丹", "family": "百合科", "genus": "百合属", "color": "橙红带紫黑斑点", "season": "夏季", "cat": "bulb",
                      "desc": "百合属多年生球根花卉，花瓣强烈反卷并密布紫黑色斑点，形似虎皮，故名虎皮百合。"},
    "moon orchid": {"cn": "月光兰", "family": "兰科", "genus": "", "color": "白、淡黄", "season": "春夏", "cat": "orchid",
                      "desc": "花形奇特、色彩素雅的兰科植物，常被称作“月光兰”，花瓣细长开展，观赏价值高。"},
    "bird of paradise": {"cn": "鹤望兰", "family": "鹤望兰科", "genus": "鹤望兰属", "color": "橙黄配蓝", "season": "全年（盛花春秋）", "cat": "perennial",
                      "desc": "花序形似昂首的仙鹤，橙色萼片与蓝色花瓣对比强烈，是著名的热带切花。"},
    "monkshood": {"cn": "乌头", "family": "毛茛科", "genus": "乌头属", "color": "深蓝紫", "season": "夏秋季", "cat": "wildflower",
                      "desc": "花冠呈盔状，深蓝紫色成串开放；全株含乌头碱，有剧毒，仅作观赏。"},
    "globe thistle": {"cn": "蓝刺头", "family": "菊科", "genus": "蓝刺头属", "color": "钢蓝、银灰", "season": "夏季", "cat": "wildflower",
                      "desc": "花序密集成球形，苞片先端硬化成刺，呈独特的钢蓝色，是优良的干花与蜜源植物。"},
    "snapdragon": {"cn": "金鱼草", "family": "车前科", "genus": "金鱼草属", "color": "红、黄、白、紫", "season": "春秋", "cat": "annual",
                      "desc": "唇形花冠形似金鱼嘴，用手轻捏两侧花瓣会像嘴一样张合，花色繁多、花序挺立。"},
    "colt's foot": {"cn": "款冬", "family": "菊科", "genus": "款冬属", "color": "亮黄", "season": "早春", "cat": "wildflower",
                      "desc": "先花后叶，早春从根茎直接抽出黄色头状花序；叶片大而圆，形似马蹄，故名款冬。"},
    "king protea": {"cn": "帝王花", "family": "山龙眼科", "genus": "帝王花属", "color": "粉红、乳白", "season": "春夏", "cat": "shrub",
                      "desc": "南非国花。头状花序巨大，外围苞片开展如王冠，是山龙眼科最具代表性的种类。"},
    "spear thistle": {"cn": "矛叶蓟", "family": "菊科", "genus": "蓟属", "color": "紫红", "season": "夏秋季", "cat": "wildflower",
                      "desc": "二年生草本，茎叶具尖锐的刺，紫色头状花序上密生细管状花，是苏格兰的象征植物。"},
    "yellow iris": {"cn": "黄菖蒲", "family": "鸢尾科", "genus": "鸢尾属", "color": "亮黄", "season": "春夏季", "cat": "perennial",
                      "desc": "鸢尾属水生或湿生多年生草本，剑形叶挺立，黄色花朵下垂，常植于池畔水边。"},
    "globe-flower": {"cn": "金莲花", "family": "毛茛科", "genus": "金莲花属", "color": "金黄", "season": "夏季", "cat": "perennial",
                      "desc": "花冠呈半球形的金黄色“球”，花瓣状萼片多层包裹，高山草甸的代表性花卉。"},
    "purple coneflower": {"cn": "紫锥菊", "family": "菊科", "genus": "松果菊属", "color": "紫红", "season": "夏季", "cat": "perennial",
                      "desc": "头状花序中央的管状花隆起成圆锥状，舌状花紫红且常反折下垂，兼具观赏与药用价值。"},
    "peruvian lily": {"cn": "六出花", "family": "六出花科", "genus": "六出花属", "color": "橙、粉、黄", "season": "春夏季", "cat": "perennial",
                      "desc": "原产南美，花被片六枚并带深色条纹，花期长、瓶插寿命长，是重要切花。"},
    "balloon flower": {"cn": "桔梗", "family": "桔梗科", "genus": "桔梗属", "color": "蓝紫、白", "season": "夏秋季", "cat": "perennial",
                      "desc": "花蕾期花冠膨大如气球，开放后呈五裂的星形钟状花，根可入药或腌食。"},
    "giant white arum lily": {"cn": "马蹄莲", "family": "天南星科", "genus": "马蹄莲属", "color": "纯白", "season": "冬春季", "cat": "perennial",
                      "desc": "佛焰苞洁白呈漏斗状，中央为黄色肉穗花序；原产南非湿地，是最经典的白色切花之一。"},
    "fire lily": {"cn": "火焰百合", "family": "百合科", "genus": "", "color": "橙红", "season": "夏季", "cat": "bulb",
                      "desc": "花色鲜艳如火焰的球根花卉，花瓣狭长反卷，色彩醒目，常用于花境与切花。"},
    "pincushion flower": {"cn": "蓝盆花", "family": "忍冬科", "genus": "蓝盆花属", "color": "淡蓝、紫", "season": "春至秋", "cat": "perennial",
                      "desc": "头状花序外围舌状花开展、中央隆起如针插，淡蓝色调柔和，花期极长。"},
    "fritillary": {"cn": "贝母", "family": "百合科", "genus": "贝母属", "color": "紫、黄、橙", "season": "春季", "cat": "bulb",
                      "desc": "鳞茎由肉质鳞片抱合而成，花冠钟形并常带棋盘状斑纹，多种贝母为传统药材。"},
    "red ginger": {"cn": "红姜花", "family": "姜科", "genus": "姜花属", "color": "鲜红", "season": "夏秋季", "cat": "bulb",
                      "desc": "苞片鲜红蜡质、层层排列成穗状，真正的花小而白；热带著名切花，花后不易变色。"},
    "grape hyacinth": {"cn": "葡萄风信子", "family": "天门冬科", "genus": "蓝壶花属", "color": "蓝紫、白", "season": "早春", "cat": "bulb",
                      "desc": "小花密集成串，形似一串葡萄，蓝紫色最为常见，是春季花坛与盆栽的小型球根花卉。"},
    "corn poppy": {"cn": "虞美人", "family": "罂粟科", "genus": "罂粟属", "color": "鲜红（亦有粉白）", "season": "春夏季", "cat": "wildflower",
                      "desc": "花瓣薄如绢纸、鲜红醒目，花心具深色斑；田野杂草型花卉，是欧洲常见的红色花海主角。"},
    "prince of wales feathers": {"cn": "威尔士亲王羽", "family": "", "genus": "", "color": "紫红", "season": "夏秋季", "cat": "annual",
                      "desc": "花序呈挺拔的羽毛状穗，颜色浓艳，形态独特，常用于花境背景与干花材料。"},
    "stemless gentian": {"cn": "无茎龙胆", "family": "龙胆科", "genus": "龙胆属", "color": "深蓝", "season": "晚春至初夏", "cat": "wildflower",
                      "desc": "高山型龙胆，花冠呈漏斗状或钟状，深蓝色极为纯正，紧贴地面开放，是阿尔卑斯山代表性花卉。"},
    "artichoke": {"cn": "朝鲜蓟", "family": "菊科", "genus": "菜蓟属", "color": "紫红", "season": "夏季", "cat": "perennial",
                      "desc": "大型蓟类，未开放的花苞可食用；花苞开放后呈紫色管状花簇，兼具食用与观赏价值。"},
    "sweet william": {"cn": "须苞石竹", "family": "石竹科", "genus": "石竹属", "color": "红、粉、白、复色", "season": "春夏季", "cat": "perennial",
                      "desc": "花小而密集成平顶伞房花序，花瓣常带环状色斑，花期长、耐寒，是花坛常用材料。"},
    "carnation": {"cn": "香石竹（康乃馨）", "family": "石竹科", "genus": "石竹属", "color": "粉、红、白、黄", "season": "全年（盛花春夏季）", "cat": "perennial",
                      "desc": "花瓣边缘具锯齿状皱褶，具淡香，是世界四大切花之一，也是母亲节的象征花卉。"},
    "garden phlox": {"cn": "宿根福禄考", "family": "花荵科", "genus": "福禄考属", "color": "粉紫、白、红", "season": "夏季", "cat": "perennial",
                      "desc": "多年生宿根草本，圆锥花序密生小花，色彩明快、具淡香，是夏花境的主力。"},
    "love in the mist": {"cn": "黑种草", "family": "毛茛科", "genus": "黑种草属", "color": "淡蓝", "season": "春夏季", "cat": "wildflower",
                      "desc": "花被片淡蓝，外围被细裂如羽毛的苞片环绕，仿佛笼罩在薄雾中，故得名“雾中情人”。"},
    "mexican aster": {"cn": "大波斯菊", "family": "菊科", "genus": "秋英属", "color": "粉、白、紫红", "season": "夏秋季", "cat": "annual",
                      "desc": "一年生草本，茎细叶纤，头状花序轻盈，随风摇曳，是常见的野花景观与自播花卉。"},
    "alpine sea holly": {"cn": "高山刺芹", "family": "伞形科", "genus": "刺芹属", "color": "银蓝", "season": "夏季", "cat": "wildflower",
                      "desc": "苞片坚硬具刺、呈银蓝色，形似蓟但实为伞形科植物，是高级干花与花艺材料。"},
    "ruby-lipped cattleya": {"cn": "红唇卡特兰", "family": "兰科", "genus": "卡特兰属", "color": "粉紫配深红唇瓣", "season": "秋季", "cat": "orchid",
                      "desc": "附生兰，花朵大而华丽，唇瓣呈浓艳的深红紫色，是卡特兰属中最著名的“洋兰”之一。"},
    "cape flower": {"cn": "好望角花", "family": "", "genus": "", "color": "粉、紫、白", "season": "春夏季", "cat": "perennial",
                      "desc": "原产南非好望角地区的花卉，花形似雏菊或百合，色彩明丽，适应干燥与强光环境。"},
    "great masterwort": {"cn": "大星芹", "family": "伞形科", "genus": "星芹属", "color": "粉白、淡绿", "season": "初夏", "cat": "perennial",
                      "desc": "伞形花序外围苞片呈星状开展，中央小花密集，整体如繁星，是花艺中常用的填充花材。"},
    "siam tulip": {"cn": "暹罗郁金香", "family": "姜科", "genus": "姜黄属", "color": "粉紫", "season": "夏季", "cat": "bulb",
                      "desc": "虽名“郁金香”，实为姜科植物，粉紫色苞片叠成球状花序，原产东南亚，是热门的热带切花。"},
    "lenten rose": {"cn": "铁筷子", "family": "毛茛科", "genus": "铁筷子属", "color": "紫、粉、白、绿", "season": "冬末至早春", "cat": "shrub",
                      "desc": "常绿宿根，花期极早，常在四旬斋前后开放；花萼瓣化且持久，耐寒耐阴，是冬季少花的珍贵来源。"},
    "barbeton daisy": {"cn": "非洲雏菊", "family": "菊科", "genus": "蓝眼菊属", "color": "白、紫、橙", "season": "春至秋", "cat": "annual",
                      "desc": "头状花序舌状花光滑，部分品种管状花呈深蓝色“眼”，花期长、耐旱，是花坛与吊篮常用花。"},
    "daffodil": {"cn": "水仙", "family": "石蒜科", "genus": "水仙属", "color": "黄、白", "season": "冬末至早春", "cat": "bulb",
                      "desc": "球根花卉，具杯状副花冠与六枚花被片，冬春开放、清香宜人，是中国传统年宵花。"},
    "sword lily": {"cn": "唐菖蒲", "family": "鸢尾科", "genus": "唐菖蒲属", "color": "红、黄、粉、紫", "season": "夏秋季", "cat": "bulb",
                      "desc": "球茎花卉，叶片剑形，穗状花序自下而上依次开放，是世界四大切花之一，俗称剑兰。"},
    "poinsettia": {"cn": "一品红", "family": "大戟科", "genus": "大戟属", "color": "鲜红（苞片）", "season": "冬季", "cat": "shrub",
                      "desc": "真正的花很小，醒目的红色其实是顶端的变态叶（苞片）；是圣诞节的标志性盆栽。"},
    "bolero deep blue": {"cn": "深蓝波莱罗", "family": "", "genus": "", "color": "深蓝紫", "season": "春夏季", "cat": "perennial",
                      "desc": "花被片呈深蓝紫色、中央具深色花心，色调浓郁沉稳，多用于花境配色。"},
    "wallflower": {"cn": "桂竹香", "family": "十字花科", "genus": "糖芥属", "color": "橙、黄、红棕", "season": "春季", "cat": "wildflower",
                      "desc": "十字形花冠，橙黄至红棕色，具浓郁甜香，常丛生于墙垣石缝，故名墙头花。"},
    "marigold": {"cn": "万寿菊", "family": "菊科", "genus": "万寿菊属", "color": "橙、金黄", "season": "夏秋季", "cat": "annual",
                      "desc": "头状花序繁密、橙黄色浓烈，气味特殊，花期极长且管理粗放，是全球最常见的花坛花卉之一。"},
    "buttercup": {"cn": "毛茛", "family": "毛茛科", "genus": "毛茛属", "color": "亮黄", "season": "春季", "cat": "wildflower",
                      "desc": "花瓣表面光滑如涂蜡，能反射光线形成“金光”，是温带草地最常见的野生黄花。"},
    "oxeye daisy": {"cn": "滨菊", "family": "菊科", "genus": "滨菊属", "color": "白瓣黄心", "season": "初夏", "cat": "wildflower",
                      "desc": "白色舌状花配黄色花心，形态经典如雏菊，常成片开放形成白色花海。"},
    "common dandelion": {"cn": "蒲公英", "family": "菊科", "genus": "蒲公英属", "color": "金黄", "season": "春至秋", "cat": "wildflower",
                      "desc": "头状花序全为舌状花，果序为白色冠毛球，可借风力传播；嫩叶可食，全草入药。"},
    "petunia": {"cn": "矮牵牛", "family": "茄科", "genus": "碧冬茄属", "color": "紫、粉、红、白、复色", "season": "春至秋", "cat": "annual",
                      "desc": "喇叭状花冠，花色极其丰富且有单双瓣与斑纹品种，是吊篮与花坛用量最大的草花之一。"},
    "wild pansy": {"cn": "三色堇", "family": "堇菜科", "genus": "堇菜属", "color": "紫、黄、白三色", "season": "春至秋", "cat": "wildflower",
                      "desc": "花冠具紫、黄、白三色，中央常有深色斑，形似猫脸，耐寒性强，是早春花坛主力。"},
    "primula": {"cn": "报春花", "family": "报春花科", "genus": "报春花属", "color": "粉、紫、红、白", "season": "早春", "cat": "perennial",
                      "desc": "伞形或层状花序，花色明艳，是春季最早开放的花卉之一，常作年宵与早春花坛用花。"},
    "sunflower": {"cn": "向日葵", "family": "菊科", "genus": "向日葵属", "color": "金黄", "season": "夏秋季", "cat": "annual",
                      "desc": "大型头状花序，花盘随光转动；种子可食可榨油，是最具辨识度的向阳花卉。"},
    "pelargonium": {"cn": "天竺葵", "family": "牻牛儿苗科", "genus": "天竺葵属", "color": "红、粉、白、橙", "season": "春至秋", "cat": "perennial",
                      "desc": "伞形花序成团开放，叶片常带特殊气味（玫瑰香、柠檬香等），是阳台与窗台最常见的盆栽花卉。"},
    "bishop of llandaff": {"cn": "主教红大丽花", "family": "菊科", "genus": "大丽花属", "color": "深红", "season": "夏秋季", "cat": "annual",
                      "desc": "大丽花经典品种，深红色花朵配深紫黑色叶片，色彩对比强烈，是花境中的视觉焦点。"},
    "gaura": {"cn": "山桃草", "family": "柳叶菜科", "genus": "山桃草属", "color": "白、粉", "season": "夏秋季", "cat": "wildflower",
                      "desc": "小花具长柄，开放时如白色蝴蝶随风飞舞，故英文又名“whirling butterflies”，极耐旱。"},
    "geranium": {"cn": "老鹳草", "family": "牻牛儿苗科", "genus": "老鹳草属", "color": "紫、粉、蓝", "season": "春夏", "cat": "wildflower",
                      "desc": "花冠五瓣，常带深色脉纹，蒴果形似鹤嘴（老鹳嘴）；园艺上多为耐寒的宿根地被植物。"},
    "orange dahlia": {"cn": "橙色大丽花", "family": "菊科", "genus": "大丽花属", "color": "橙色", "season": "夏秋季", "cat": "annual",
                      "desc": "块根花卉，花形与花色极其丰富，橙色品种明快热烈，花期可延续至霜降。"},
    "pink-yellow dahlia": {"cn": "粉黄大丽花", "family": "菊科", "genus": "大丽花属", "color": "粉黄复色", "season": "夏秋季", "cat": "annual",
                      "desc": "大丽花复色品种，粉与黄渐变过渡，花瓣层叠饱满，是切花与花境的高价值材料。"},
    "cautleya spicata": {"cn": "斑叶距药姜", "family": "姜科", "genus": "距药姜属", "color": "黄配红苞", "season": "夏秋季", "cat": "bulb",
                      "desc": "姜科多年生，花序穗状，黄色小花自红色苞片中伸出，叶片具紫红色叶背，观花观叶兼宜。"},
    "japanese anemone": {"cn": "秋牡丹", "family": "毛茛科", "genus": "银莲花属", "color": "粉、白", "season": "秋季", "cat": "perennial",
                      "desc": "秋季开花的宿根草本，花被片开展，中央密生黄色雄蕊，花后结出白色绒毛状聚合果。"},
    "black-eyed susan": {"cn": "黑心菊", "family": "菊科", "genus": "金光菊属", "color": "金黄配黑心", "season": "夏秋季", "cat": "perennial",
                      "desc": "金黄色的舌状花围绕深褐色隆起的管状花心，色彩对比强烈，耐旱耐热、花期长。"},
    "silverbush": {"cn": "银叶旋花", "family": "旋花科", "genus": "旋花属", "color": "白", "season": "春至秋", "cat": "wildflower",
                      "desc": "矮小常绿灌木，叶片密被银色绢毛，白色漏斗状花带黄心，是优秀的地被与岩石园植物。"},
    "californian poppy": {"cn": "花菱草", "family": "罂粟科", "genus": "花菱草属", "color": "橙黄", "season": "春夏季", "cat": "wildflower",
                      "desc": "花瓣薄而光滑，阳光下充分开展、阴天或夜晚闭合，原产加州，极耐旱、能自播。"},
    "osteospermum": {"cn": "蓝眼菊", "family": "菊科", "genus": "蓝眼菊属", "color": "白、紫、橙", "season": "春至秋", "cat": "annual",
                      "desc": "舌状花光滑，管状花常呈金属蓝或深紫色“眼”，花色明快，是春季花坛与盆栽的重要材料。"},
    "spring crocus": {"cn": "春番红花", "family": "鸢尾科", "genus": "番红花属", "color": "紫、白、黄", "season": "早春", "cat": "bulb",
                      "desc": "球茎花卉，早春自地面抽出漏斗状小花，是最早报春的球根之一；柱头可作香料与药材。"},
    "bearded iris": {"cn": "有髯鸢尾", "family": "鸢尾科", "genus": "鸢尾属", "color": "紫、蓝、黄、褐", "season": "晚春至初夏", "cat": "perennial",
                      "desc": "下垂的外花被片中脉上具毛刷状“髯”，花色丰富华丽，是鸢尾属园艺品种最多的一类。"},
    "windflower": {"cn": "银莲花", "family": "毛茛科", "genus": "银莲花属", "color": "白、蓝、红", "season": "春季", "cat": "wildflower",
                      "desc": "花被片薄而开展，风过时随风摇曳，故名风花；多为春季开花的低矮宿根或球根花卉。"},
    "tree poppy": {"cn": "金罂粟", "family": "罂粟科", "genus": "金罂粟属", "color": "亮黄", "season": "夏秋季", "cat": "shrub",
                      "desc": "常绿小灌木，花瓣薄如纸、呈明亮的黄色，花量大且花期长，耐旱喜光。"},
    "gazania": {"cn": "勋章菊", "family": "菊科", "genus": "勋章菊属", "color": "黄橙配深色环", "season": "春至秋", "cat": "annual",
                      "desc": "舌状花基部具深色环纹，形似勋章；喜光，花朵在强光下充分开放、傍晚闭合。"},
    "azalea": {"cn": "杜鹃", "family": "杜鹃花科", "genus": "杜鹃花属", "color": "红、粉、紫、白", "season": "春季", "cat": "shrub",
                      "desc": "落叶或半常绿灌木，花冠漏斗状、常 2—5 朵聚生枝顶，盛开时花团锦簇，是中国传统名花。"},
    "water lily": {"cn": "睡莲", "family": "睡莲科", "genus": "睡莲属", "color": "白、粉、黄、蓝", "season": "夏季", "cat": "aquatic",
                      "desc": "水生多年生，叶片浮于水面并具深裂缺刻，花朵多白天开放、傍晚闭合，是水景的核心植物。"},
    "rose": {"cn": "玫瑰", "family": "蔷薇科", "genus": "蔷薇属", "color": "红、粉、白、黄", "season": "春至秋", "cat": "shrub",
                      "desc": "蔷薇属灌木，花冠重瓣、芳香浓郁；观赏与香料价值极高，是世界上最主要的切花与庭院花卉。"},
    "thorn apple": {"cn": "曼陀罗", "family": "茄科", "genus": "曼陀罗属", "color": "白、淡紫", "season": "夏秋季", "cat": "wildflower",
                      "desc": "一年生草本，花冠长喇叭状，果实密生硬刺；全株有毒，仅作观赏与药用研究。"},
    "morning glory": {"cn": "牵牛花", "family": "旋花科", "genus": "牵牛属", "color": "蓝、紫、粉、白", "season": "夏秋季", "cat": "vine",
                      "desc": "一年生缠绕藤本，漏斗状花清晨开放、午后凋谢，故又名朝颜，生长迅速、能自播。"},
    "passion flower": {"cn": "西番莲", "family": "西番莲科", "genus": "西番莲属", "color": "紫、白、蓝", "season": "夏秋季", "cat": "vine",
                      "desc": "花形结构极为独特：副花冠呈放射状丝状、中央为挺立的雌雄蕊柄，是热带著名的观赏藤本。"},
    "lotus": {"cn": "荷花", "family": "莲科", "genus": "莲属", "color": "粉、白", "season": "夏季", "cat": "aquatic",
                      "desc": "水生多年生，叶片与花均挺出水面，地下茎为藕；兼具观赏、食用与文化象征意义。"},
    "toad lily": {"cn": "油点草", "family": "百合科", "genus": "油点草属", "color": "白配紫斑", "season": "夏秋季", "cat": "perennial",
                      "desc": "花被片白色并密布紫红色斑点（形似蟾蜍皮纹，故名），是秋季少花的耐阴宿根花卉。"},
    "anthurium": {"cn": "火鹤花", "family": "天南星科", "genus": "花烛属", "color": "鲜红配黄肉穗", "season": "全年", "cat": "perennial",
                      "desc": "佛焰苞心形、蜡质有光泽，中央肉穗花序挺立；单花寿命极长，是高档热带切花与盆栽。"},
    "frangipani": {"cn": "鸡蛋花", "family": "夹竹桃科", "genus": "鸡蛋花属", "color": "白配黄心（亦有粉红）", "season": "夏秋季", "cat": "shrub",
                      "desc": "落叶小乔木，五枚花瓣螺旋状排列，白花黄心，香气浓郁，是热带地区常见的庭园树种。"},
    "clematis": {"cn": "铁线莲", "family": "毛茛科", "genus": "铁线莲属", "color": "紫、粉、白、红", "season": "春至秋", "cat": "vine",
                      "desc": "藤本皇后。花大色艳，部分品种花后结出银丝状羽状果序，观赏期长，需依附支架生长。"},
    "hibiscus": {"cn": "朱槿", "family": "锦葵科", "genus": "木槿属", "color": "红、粉、黄、白", "season": "夏秋季", "cat": "shrub",
                      "desc": "常绿灌木，花冠大而呈喇叭状，雄蕊柱显著伸出，单花朝开暮落但花期连续不断。"},
    "columbine": {"cn": "耧斗菜", "family": "毛茛科", "genus": "耧斗菜属", "color": "蓝、紫、红、黄", "season": "春夏季", "cat": "wildflower",
                      "desc": "花瓣基部延伸成距，形似耧斗，花形别致、自然飘逸，是林下与花境的优良宿根花卉。"},
    "desert-rose": {"cn": "沙漠玫瑰", "family": "夹竹桃科", "genus": "天宝花属", "color": "粉红、红、白", "season": "春至秋", "cat": "succulent",
                      "desc": "肉质茎膨大如瓶、善于储水，花朵鲜艳如玫瑰；原产干旱地区，喜高温强光、极耐旱。"},
    "tree mallow": {"cn": "花葵", "family": "锦葵科", "genus": "花葵属", "color": "紫红、粉", "season": "春夏", "cat": "vine",
                      "desc": "半灌木或大型草本，花冠五瓣、具放射状脉纹，生长强健，适合花境背景与海岸绿化。"},
    "magnolia": {"cn": "玉兰", "family": "木兰科", "genus": "木兰属", "color": "白、粉、紫", "season": "早春", "cat": "shrub",
                      "desc": "落叶乔木或灌木，花大而芳香，多在叶前开放，是早春最壮观的开花树木之一。"},
    "cyclamen": {"cn": "仙客来", "family": "报春花科", "genus": "仙客来属", "color": "红、粉、白", "season": "冬春季", "cat": "bulb",
                      "desc": "块茎花卉，花瓣强烈上翻形似兔耳，叶面常有银白色斑纹；是冬春最重要的室内盆栽之一。"},
    "watercress": {"cn": "水田芥（西洋菜）", "family": "十字花科", "genus": "豆瓣菜属", "color": "白", "season": "春至秋", "cat": "aquatic",
                      "desc": "水生或湿生草本，十字形小白花；嫩茎叶可食，是常见的蔬菜，也适合作水景边缘绿化。"},
    "canna lily": {"cn": "美人蕉", "family": "美人蕉科", "genus": "美人蕉属", "color": "红、黄、橙、复色", "season": "夏秋季", "cat": "bulb",
                      "desc": "根茎粗壮，叶片宽大如芭蕉，花序顶生且花色浓艳，花期长，是热带风情的花坛主角。"},
    "hippeastrum": {"cn": "朱顶红", "family": "石蒜科", "genus": "朱顶红属", "color": "红、粉、白、条纹", "season": "冬春季", "cat": "bulb",
                      "desc": "大型球根花卉，花茎自鳞茎抽出，顶端着生 2—6 朵喇叭状大花，是极受欢迎的室内年宵花。"},
    "bee balm": {"cn": "美国薄荷", "family": "唇形科", "genus": "美国薄荷属", "color": "红、粉、紫", "season": "夏季", "cat": "perennial",
                      "desc": "唇形科宿根草本，花序呈管状花密集的头状，叶片揉碎有薄荷与柑橘香，极能吸引蜂鸟与蜜蜂。"},
    "ball moss": {"cn": "球状铁兰", "family": "凤梨科", "genus": "铁兰属", "color": "灰绿（花淡紫）", "season": "春夏季", "cat": "succulent",
                      "desc": "附生凤梨，以银灰色细叶簇生成球状，靠叶面鳞片吸收空气中的水分，不寄生宿主。"},
    "foxglove": {"cn": "毛地黄", "family": "车前科", "genus": "毛地黄属", "color": "紫、粉、白", "season": "初夏", "cat": "wildflower",
                      "desc": "二年生直立草本，钟形花自下而上沿花茎开放，内壁常具斑点；全株有毒，是重要强心苷来源植物。"},
    "bougainvillea": {"cn": "三角梅", "family": "紫茉莉科", "genus": "叶子花属", "color": "紫红、粉、橙、白", "season": "夏秋季", "cat": "vine",
                      "desc": "真正的花很小，鲜艳的“花瓣”实为三枚苞片；攀援性强、耐旱耐热，是热带城市的标志性花木。"},
    "camellia": {"cn": "山茶", "family": "山茶科", "genus": "山茶属", "color": "红、粉、白", "season": "冬春季", "cat": "shrub",
                      "desc": "常绿灌木或小乔木，叶片革质有光泽，花大且花瓣层叠，是中国传统十大名花之一。"},
    "mallow": {"cn": "锦葵", "family": "锦葵科", "genus": "锦葵属", "color": "紫红、粉", "season": "夏秋季", "cat": "wildflower",
                      "desc": "直立草本，花冠五瓣、具深紫色脉纹，花小而多，管理粗放，是欧洲乡野常见的锦葵科植物。"},
    "mexican petunia": {"cn": "蓝花草", "family": "爵床科", "genus": "芦莉草属", "color": "蓝紫、粉", "season": "夏秋季", "cat": "shrub",
                      "desc": "花冠漏斗状、边缘微皱，蓝紫色清新，花期长而耐热耐旱，是南方常见的花境与绿篱材料。"},
    "bromelia": {"cn": "凤梨（观赏凤梨）", "family": "凤梨科", "genus": "凤梨属", "color": "红、橙、黄", "season": "春夏", "cat": "succulent",
                      "desc": "叶片莲座状排列，中央叶丛在花期变为鲜艳的红橙色（观赏部位为苞片），耐阴，是室内观叶观花植物。"},
    "blanket flower": {"cn": "天人菊", "family": "菊科", "genus": "天人菊属", "color": "红配黄边", "season": "夏秋季", "cat": "annual",
                      "desc": "舌状花基部红、先端黄，如覆盖地面的彩毯（故英文名“毯子花”），极耐旱耐热、花期长。"},
    "trumpet creeper": {"cn": "凌霄花", "family": "紫葳科", "genus": "凌霄属", "color": "橙红", "season": "夏秋季", "cat": "vine",
                      "desc": "落叶木质藤本，以气生根攀附墙面树干，喇叭状橙红花朵成串开放，是盛夏重要的垂直绿化植物。"},
    "blackberry lily": {"cn": "射干", "family": "鸢尾科", "genus": "射干属", "color": "橙红带紫斑", "season": "夏秋季", "cat": "perennial",
                      "desc": "鸢尾科宿根草本，花被片反卷并密布紫红斑点，果实成熟开裂露出黑色种子似黑莓，故名黑莓百合。"},
}


def build(export_thumbs: bool = False, strict: bool = False) -> int:
    names = get_class_names()
    print(f"[build_flowers] 目标类别数：{len(names)}")

    missing = [n for n in names if n not in SPECIES]
    extra = [k for k in SPECIES if k not in names]
    if missing or extra:
        print(f"✗ 物种表与类别名不匹配：缺失 {len(missing)} 项，多余 {len(extra)} 项")
        for n in missing[:10]:
            print(f"   缺失: {n!r}")
        for n in extra[:10]:
            print(f"   多余: {n!r}")
        return 1
    print(f"✓ 物种表覆盖全部 {len(names)} 个类别，无缺失、无多余")

    entry_builders: list[tuple[str, str, dict]] = []
    id_by_class: dict[int, int] = {}

    # 缩略图映射：class_id -> 某张 train 图
    thumb_src: dict[int, Path] = {}
    if export_thumbs:
        mf = load_manifest("data/flowers-102")
        for f in mf.splits["train"]:
            thumb_src.setdefault(mf.labels[f], mf.root / mf.image_dir / f)

    if export_thumbs:
        from PIL import Image

        # 与后端静态目录对齐：backend/app/static/florers -> 实际用 THUMB_DIR
        rel_static = "/static/flowers"
        THUMB_DIR.mkdir(parents=True, exist_ok=True)
        for cid in range(len(names)):
            src = thumb_src.get(cid)
            if src is None or not src.exists():
                print(f"⚠ class {cid} 无可用样本图，缩略图跳过")
                continue
            dst = THUMB_DIR / f"{cid:03d}.jpg"
            if not dst.exists():
                with Image.open(src) as im:
                    im.convert("RGB").thumbnail((THUMB_SIZE, THUMB_SIZE), Image.LANCZOS)
                    im.save(dst, format="JPEG", quality=85, optimize=True)
        print(f"✓ 百科缩略图已导出：{THUMB_DIR}（{len(list(THUMB_DIR.glob('*.jpg')))} 张）")

    records: list[dict] = []
    field_missing = 0
    for cid, name_en in enumerate(names):
        sp = SPECIES[name_en]
        cat = sp["cat"]
        if cat not in CARE_BY_CATEGORY:
            print(f"✗ {name_en!r} 的类别 {cat!r} 不在 CARE_BY_CATEGORY 中")
            return 1
        care = CARE_BY_CATEGORY[cat]
        for key in ("cn", "family", "genus", "color", "season", "desc"):
            if not sp.get(key):
                field_missing += 1
        record = {
            "class_id": cid,
            "name_en": name_en,
            "name_cn": sp["cn"],
            "category": cat,
            "category_cn": care["category_cn"],
            "family": sp["family"],
            "genus": sp["genus"],
            "color": sp["color"],
            "bloom_season": sp["season"],
            "light": care["light"],
            "watering": care["watering"],
            "soil": care["soil"],
            "propagation": care["propagation"],
            "description": sp["desc"],
            "care_tips": care["care_tips"],
            "sample_image": f"/static/flowers/{cid:03d}.jpg" if export_thumbs else "",
            "curated": "basic",
        }
        records.append(record)

    save_json(records, OUT_JSON)
    print(f"✓ 已写出 {OUT_JSON}（{len(records)} 条）")
    print(f"  name_cn 非空：{sum(1 for r in records if r['name_cn'])}/{len(records)}")
    print(f"  family 非空 ：{sum(1 for r in records if r['family'])}/{len(records)}")
    print(f"  genus  非空 ：{sum(1 for r in records if r['genus'])}/{len(records)}")
    if field_missing:
        print(f"  ℹ 共 {field_missing} 个字段留空（前端显示“暂无资料”），不影响功能")

    if strict and field_missing:
        print("✗ --strict 模式下不允许留空字段")
        return 1
    return 0


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="构建花卉百科 flowers.json（102 类）")
    ap.add_argument("--export-thumbs", action="store_true", dest="export_thumbs",
                    help="同时从数据集导出 102 张百科缩略图")
    ap.add_argument("--strict", action="store_true", help="有留空字段则失败")
    args = ap.parse_args(argv)
    return build(export_thumbs=args.export_thumbs, strict=args.strict)


if __name__ == "__main__":
    sys.exit(main())
