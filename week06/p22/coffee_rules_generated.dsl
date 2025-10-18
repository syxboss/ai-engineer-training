WORKFLOW "智能咖啡制作系统" VERSION 1.2

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

# 水温控制 - 支持自定义温度
NODE heat_water TYPE action
  DO TURN_ON heater
  DO SET target_temp = 90°C
  DESCRIPTION "加热水到目标温度90°C"

NODE check_temp TYPE condition
  WHEN water.temp >= 89°C
  DESCRIPTION "检查水温是否达标"

NODE temp_ready TYPE action
  DO SEND "水温已达到90°C，可以开始制作咖啡" TO display
  DESCRIPTION "水温就绪通知"

# 咖啡制作 - 支持自定义萃取时间和强度
NODE prepare_coffee TYPE action
  DO SET extraction_strength = "medium"
  DO SEND "准备medium萃取" TO display
  DESCRIPTION "设置萃取强度为medium"

NODE make_coffee TYPE action
  DO START brewing
  DO WAIT 20s
  DO STOP brewing
  DESCRIPTION "制作咖啡，萃取时间20秒"

NODE coffee_ready TYPE action
  DO TURN_OFF heater
  DO SEND "咖啡制作完成！温度90°C，萃取20秒，medium萃取" TO display
  DESCRIPTION "咖啡制作完成"

# 主流程
EDGE start -> check_water
EDGE check_water -> heat_water
EDGE check_water -> add_water CONDITION water.level < 300ml
EDGE add_water -> heat_water
EDGE heat_water -> check_temp
EDGE check_temp -> temp_ready
EDGE temp_ready -> prepare_coffee
EDGE prepare_coffee -> make_coffee
EDGE make_coffee -> coffee_ready