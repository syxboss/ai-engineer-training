# logistics_mcp_server.py
from mcp.server.fastmcp import FastMCP

mcp = FastMCP("Logistics")

@mcp.tool()
def get_package_status(tracking_id: str) -> str:
    """Get the current status of a package by tracking ID."""
    statuses = {
        "LGT123456": "已发货，在途中",
        "LGT789012": "本地分拣中心，待派送",
        "LGT345678": "已签收",
        "LGT000000": "无效单号，请检查"
    }
    return statuses.get(tracking_id, "未找到该包裹信息")

@mcp.tool()
def calculate_shipping_cost(weight_kg: float, distance_km: float) -> str:
    """Calculate shipping cost based on weight and distance."""
    base_rate = 5.0
    per_kg_rate = 2.0
    per_km_rate = 0.01
    cost = base_rate + (weight_kg * per_kg_rate) + (distance_km * per_km_rate)
    return f"运费估算：{round(cost, 2)} 元"

@mcp.tool()
def estimate_delivery_time(distance_km: float) -> str:
    """Estimate delivery time in hours based on distance."""
    avg_speed_km_per_hour = 40
    time_hours = distance_km / avg_speed_km_per_hour
    days = int(time_hours // 24)
    hours = round(time_hours % 24)
    return f"预计送达时间：{days} 天 {hours} 小时"

if __name__ == "__main__":
    mcp.run(transport="streamable-http")