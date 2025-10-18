WORKFLOW "智能咖啡制作系统" VERSION 1.1

# 初始节点
NODE start TYPE initial
  DESCRIPTION "系统启动"

# 检查水位
NODE check_water TYPE condition
  WHEN water.level >= 300ml
  DESCRIPTION "检查水箱水位"

NODE add_water TYPE action
  DO ALERT "请添加水"
  DESCRIPTION "提醒添加水"

# 水温控制 - 核心功能
NODE heat_water TYPE action
  DO TURN_ON heater
  DO SET target_temp = 93°C
  DESCRIPTION "加热水到目标温度"

NODE check_temp TYPE condition
  WHEN water.temp >= 93°C
  DESCRIPTION "检查水温是否达标"

NODE temp_ready TYPE action
  DO SEND "水温已达到93°C，可以开始制作咖啡" TO display
  DESCRIPTION "水温就绪通知"

# 简单的咖啡制作
NODE make_coffee TYPE action
  DO START brewing
  DO WAIT 20s
  DO STOP brewing
  DESCRIPTION "制作咖啡"

NODE coffee_ready TYPE action
  DO TURN_OFF heater
  DO SEND "咖啡制作完成！" TO display
  DESCRIPTION "咖啡制作完成"

# 主流程
EDGE start -> check_water
EDGE check_water -> heat_water
EDGE check_water -> add_water CONDITION water.level < 300ml
EDGE add_water -> heat_water
EDGE heat_water -> check_temp
EDGE check_temp -> temp_ready
EDGE temp_ready -> make_coffee
EDGE make_coffee -> coffee_ready