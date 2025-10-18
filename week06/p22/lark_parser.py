from lark import Lark, Transformer

class CoffeeTransformer(Transformer):
    def workflow(self, items):
        return {"workflow_name": items[0], "version": items[1], "body": items[2:]}

    def node(self, items):
        return {"node_name": items[0], "node_type": items[1], "node_data": items[2:]}

    def edge(self, items):
        return {"source": items[0], "target": items[1], "condition": items[2] if len(items) > 2 else None}

    def edge_condition(self, items):
        return items[0]

    def condition(self, items):
        return {"sensor": items[0], "op": items[1], "value": items[2], "unit": items[3] if len(items) > 3 else None}

    def action(self, items):
        return items[0]

    def wait_action(self, items):
        return {"action_type": "wait", "duration": items[0], "unit": items[1] if len(items) > 1 else None}

    def start_action(self, items):
        return {"action_type": "start", "process": items[0]}

    def stop_action(self, items):
        return {"action_type": "stop", "process": items[0]}

    def turn_on_action(self, items):
        return {"action_type": "turn_on", "device": items[0]}

    def turn_off_action(self, items):
        return {"action_type": "turn_off", "device": items[0]}

    def send_action(self, items):
        return {"action_type": "send", "message": items[0], "recipient": items[1]}

    def alert_action(self, items):
        return {"action_type": "alert", "message": items[0]}

    def parameter_action(self, items):
        return {"action_type": "parameter", "parameter": items[0], "value": items[1], "unit": items[2] if len(items) > 2 else None}

    def __default__(self, data, children, meta):
        return children[0] if children else None

    def ID(self, items):
        return items.value

    def STRING(self, items):
        return items.value

    def NUMBER(self, items):
        return items.value

    def UNIT(self, items):
        return items.value

    def SENSOR_PATH(self, items):
        return items.value

    def NODE_TYPE(self, items):
        return items.value

    def COMP(self, items):
        return items.value

def parse(dsl_code):
    with open("coffee_dsl.lark", "r", encoding="utf-8") as f:
        grammar = f.read()
    
    parser = Lark(grammar, start='workflow')
    tree = parser.parse(dsl_code)
    return CoffeeTransformer().transform(tree)

if __name__ == "__main__":
    with open("coffee_rules.dsl", "r", encoding="utf-8") as f:
        dsl_code = f.read()
    
    try:
        result = parse(dsl_code)
        import json
        print(json.dumps(result, indent=2, ensure_ascii=False))
    except Exception as e:
        print(f"DSL parsing error: {e}")