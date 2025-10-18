grammar CoffeeMachine;

// 主工作流规则
workflow: 'WORKFLOW' STRING 'VERSION' NUMBER NEWLINE (node | edge)+;

// 节点定义
node: 'NODE' ID 'TYPE' nodeType NEWLINE nodeData*;
nodeType: 'initial' | 'action' | 'condition';

// 节点数据
nodeData: 'DO' action NEWLINE 
        | 'WHEN' condition NEWLINE
        | 'DESCRIPTION' STRING NEWLINE;

// 边定义
edge: 'EDGE' sourceEdge '->' targetEdge edgeCondition? NEWLINE;
sourceEdge: ID;
targetEdge: ID;
edgeCondition: 'CONDITION' condition;

// 条件表达式
condition: sensorPath comp NUMBER unit?;
sensorPath: ID ('.' ID)*;
comp: '>=' | '<=' | '>' | '<' | '==';

// 动作定义
action: waitAction 
      | startAction 
      | stopAction 
      | turnOnAction 
      | turnOffAction 
      | sendAction 
      | alertAction 
      | parameterAction;

waitAction: 'WAIT' NUMBER unit;
startAction: 'START' ID;
stopAction: 'STOP' ID;
turnOnAction: 'TURN_ON' ID;
turnOffAction: 'TURN_OFF' ID;
sendAction: 'SEND' STRING 'TO' ID;
alertAction: 'ALERT' STRING;
parameterAction: 'SET' ID '=' NUMBER unit?;

// 单位
unit: '°C' | 'ml' | 's';

// 词法规则
ID: [a-zA-Z_][a-zA-Z0-9_]*;
STRING: '"' (~["\r\n])* '"';
NUMBER: '-'? [0-9]+ ('.' [0-9]+)?;
NEWLINE: ('\r'? '\n' | '\r')+;
COMMENT: '#' ~[\r\n]* -> skip;
WS: [ \t]+ -> skip;