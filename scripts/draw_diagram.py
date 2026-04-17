"""
DecentralAI Architecture Diagram Generator
=============================================
Generates a clean, professional system architecture diagram.
"""

import sys
sys.path.insert(0, r'D:\pylib')

import tkinter as tk
from tkinter import Canvas, ALL
import os

# Try PIL for PNG export
try:
    from PIL import Image, ImageDraw, ImageFont
    PIL_AVAILABLE = True
except ImportError:
    PIL_AVAILABLE = False
    print("PIL not available, will display with tkinter only")

W, H = 1400, 900

# Colors
BG = '#0d1117'
PANEL_BG = '#161b22'
BORDER = '#30363d'
TEXT = '#c9d1d9'
MUTED = '#8b949e'
ACCENT = '#6366f1'
GREEN = '#22c55e'
BLUE = '#3b82f6'
PURPLE = '#a855f7'
YELLOW = '#eab308'
CYAN = '#06b6d4'
RED = '#ef4444'

def get_font(size, bold=False):
    """Try to get a good font"""
    fonts = []
    if bold:
        fonts = ['Segoe UI Semibold', 'Microsoft YaHei', 'Arial', 'Helvetica']
    else:
        fonts = ['Segoe UI', 'Microsoft YaHei', 'Arial', 'Helvetica']
    for f in fonts:
        try:
            if os.name == 'nt':
                return tk.Font(family=f, size=size, weight='bold' if bold else 'normal')
        except:
            pass
    return tk.Font(size=size)


def draw_rounded_rect(canvas, x1, y1, x2, y2, r=8, **kwargs):
    """Draw a rounded rectangle"""
    fill = kwargs.get('fill', '')
    outline = kwargs.get('outline', '')
    width = kwargs.get('width', 1)
    
    # Draw the main body
    if fill:
        canvas.create_rectangle(x1+r, y1, x2-r, y2, fill=fill, outline='')
        canvas.create_rectangle(x1, y1+r, x2, y2-r, fill=fill, outline='')
    if outline:
        canvas.create_rectangle(x1+r, y1, x2-r, y1+1, fill=outline, outline='')
        canvas.create_rectangle(x1+r, y2-1, x2-r, y2, fill=outline, outline='')
        canvas.create_rectangle(x1, y1+r, x1+1, y2-r, fill=outline, outline='')
        canvas.create_rectangle(x2-1, y1+r, x2, y2-r, fill=outline, outline='')
    
    # Arcs
    canvas.create_arc(x1, y1, x1+2*r, y1+2*r, start=90, extent=90, fill=fill or '', outline=outline or '', style='arc', width=width)
    canvas.create_arc(x2-2*r, y1, x2, y1+2*r, start=0, extent=90, fill=fill or '', outline=outline or '', style='arc', width=width)
    canvas.create_arc(x1, y2-2*r, x1+2*r, y2, start=180, extent=90, fill=fill or '', outline=outline or '', style='arc', width=width)
    canvas.create_arc(x2-2*r, y2-2*r, x2, y2, start=270, extent=90, fill=fill or '', outline=outline or '', style='arc', width=width)


def draw_arrow(canvas, x1, y1, x2, y2, color=MUTED, dashed=False, label=''):
    """Draw an arrow between two points"""
    style = 'dashed' if dashed else ''
    canvas.create_line(x1, y1, x2, y2, arrow='last', fill=color, 
                      dash=(4, 4) if dashed else None, width=1.5)
    if label:
        mx, my = (x1+x2)//2, (y1+y2)//2
        canvas.create_text(mx+6, my, text=label, fill=MUTED, font=('Segoe UI', 8))


def draw_node(canvas, x, y, w, h, title, subtitle='', color=ACCENT, icon=''):
    """Draw a component node"""
    # Shadow
    draw_rounded_rect(canvas, x+2, y+2, x+w+2, y+h+2, r=8, fill='#1a1a2e')
    # Box
    draw_rounded_rect(canvas, x, y, x+w, y+h, r=8, fill=PANEL_BG, outline=color, width=1.5)
    # Title
    canvas.create_text(x+w//2, y+20, text=title, fill=TEXT, font=('Segoe UI', 11, 'bold'), anchor='c')
    if subtitle:
        canvas.create_text(x+w//2, y+38, text=subtitle, fill=MUTED, font=('Segoe UI', 9), anchor='c')
    return x+w//2, y+h//2


def draw_layer(canvas, x, y, w, h, label, color, nodes):
    """Draw a layer with label and contained nodes"""
    # Layer background
    draw_rounded_rect(canvas, x, y, x+w, y+h, r=12, fill=BG, outline=BORDER, width=1)
    # Layer header
    draw_rounded_rect(canvas, x+4, y+4, x+140, y+32, r=6, fill=color, outline='')
    canvas.create_text(x+72, y+18, text=label, fill='white', font=('Segoe UI', 10, 'bold'), anchor='c')
    return x + 8, y + 42


def create_diagram():
    root = tk.Tk()
    root.title('DecentralAI Architecture')
    
    canvas = Canvas(root, width=W, height=H, bg=BG, highlightthickness=0)
    canvas.pack()
    
    # === TITLE ===
    canvas.create_text(W//2, 35, text='DecentralAI', fill=TEXT, font=('Segoe UI', 24, 'bold'), anchor='c')
    canvas.create_text(W//2, 60, text='去中心化异构 AI 推理网络  |  dMoE Architecture', fill=MUTED, font=('Segoe UI', 11), anchor='c')
    
    # === LAYER 0: CLIENT ===
    l0_y = 85
    l0_x, l0_y2 = draw_layer(canvas, 20, l0_y, W-40, 75, 'CLIENT LAYER', BLUE, [])
    canvas.create_text(l0_x - 20, l0_y+38, text='⟨', fill=BLUE, font=('Segoe UI', 16, 'bold'), anchor='e')
    
    # Client apps
    apps = [
        (50, l0_y2+10, 150, 58, 'OpenAI SDK', 'openai-python', BLUE),
        (220, l0_y2+10, 150, 58, 'REST API', 'curl / HTTP', BLUE),
        (390, l0_y2+10, 150, 58, 'Web UI', 'dashboard', BLUE),
        (560, l0_y2+10, 150, 58, 'CLI Tool', 'evoagent', BLUE),
    ]
    for ax, ay, aw, ah, at, asub, ac in apps:
        draw_rounded_rect(canvas, ax, ay, ax+aw, ay+ah, r=6, fill=PANEL_BG, outline=ac, width=1)
        canvas.create_text(ax+aw//2, ay+22, text=at, fill=TEXT, font=('Segoe UI', 10, 'bold'), anchor='c')
        canvas.create_text(ax+aw//2, ay+40, text=asub, fill=MUTED, font=('Segoe UI', 8), anchor='c')
    
    # Arrow down to API Gateway
    mid_x = W//2
    draw_arrow(canvas, mid_x, l0_y+75, mid_x, l0_y+95, color=BLUE)
    
    # === LAYER 1: API & GATEWAY ===
    l1_y = l0_y + 95
    l1_x, l1_y2 = draw_layer(canvas, 20, l1_y, W-40, 100, 'API & GATEWAY', PURPLE, [])
    
    gw_x = W//2 - 180
    gw_y = l1_y2 + 8
    draw_rounded_rect(canvas, gw_x, gw_y, gw_x+360, gw_y+65, r=8, fill=PANEL_BG, outline=PURPLE, width=1.5)
    canvas.create_text(gw_x+180, gw_y+18, text='API Gateway', fill=PURPLE, font=('Segoe UI', 12, 'bold'), anchor='c')
    endpoints = ['/v1/chat/completions', '/v1/completions', '/v1/models', '/health', '/status', '/dashboard']
    ep_text = '  ·  '.join(endpoints)
    canvas.create_text(gw_x+180, gw_y+42, text=ep_text, fill=MUTED, font=('Segoe UI', 8), anchor='c')
    
    # Arrow to P2P
    draw_arrow(canvas, mid_x, l1_y+100, mid_x, l1_y+120, color=PURPLE)
    
    # === LAYER 2: NETWORK ===
    l2_y = l1_y + 120
    l2_x, l2_y2 = draw_layer(canvas, 20, l2_y, W-40, 110, 'P2P NETWORK LAYER', ACCENT, [])
    
    # WebSocket nodes
    ws_x = W//2 - 260
    for i, (lx, label, sub) in enumerate([
        (W//2-580, 'node_alpha', 'L2 · Standard'),
        (W//2-350, 'node_beta', 'L1 · Light'),
        (W//2-120, 'Bootstrap', 'L4 · DataCenter'),
        (W//2+110, 'node_gamma', 'L2 · Standard'),
        (W//2+340, 'node_delta', 'L3 · Heavy'),
    ]):
        nc = [ACCENT, GREEN, YELLOW, ACCENT, PURPLE][i]
        draw_rounded_rect(canvas, lx, l2_y2+8, lx+180, l2_y2+78, r=6, fill=PANEL_BG, outline=nc, width=1)
        canvas.create_text(lx+90, l2_y2+28, text=label, fill=TEXT, font=('Segoe UI', 10, 'bold'), anchor='c')
        canvas.create_text(lx+90, l2_y2+48, text=sub, fill=MUTED, font=('Segoe UI', 8), anchor='c')
        canvas.create_text(lx+90, l2_y2+66, text='🟢 WebSocket', fill=MUTED, font=('Segoe UI', 8), anchor='c')
    
    # Connection lines between nodes
    node_centers = [W//2-490, W//2-260, W//2-30, W//2+200, W//2+430]
    for i in range(len(node_centers)-1):
        c1, c2 = node_centers[i], node_centers[i+1]
        cy = l2_y2+43
        canvas.create_line(c1+90, cy, c2-80, cy, fill=BORDER, dash=(3,3), width=1)
        canvas.create_oval(c1+90-3, cy-3, c1+90+3, cy+3, fill=ACCENT)
        canvas.create_oval(c2-80-3, cy-3, c2-80+3, cy+3, fill=ACCENT)
    
    # Arrow to core
    draw_arrow(canvas, mid_x, l2_y+110, mid_x, l2_y+130, color=ACCENT)
    
    # === LAYER 3: CORE ENGINE ===
    l3_y = l2_y + 130
    l3_h = 180
    l3_x, l3_y2 = draw_layer(canvas, 20, l3_y, W-40, l3_h, 'CORE ENGINE', GREEN, [])
    
    # Core components in 2 rows
    core_components = [
        # Row 1
        (40, l3_y2+8, 200, 70, 'Node Identity', 'Capabilities · Experts\nLevel · Reputation', '#1e3a5f', BLUE),
        (260, l3_y2+8, 200, 70, 'Router', 'Expert Selection\nDomain Routing', '#1a3a2a', GREEN),
        (480, l3_y2+8, 200, 70, 'Verifier', 'Consensus\nSyntax Check', '#3a1a2a', PURPLE),
        # Row 2
        (40, l3_y2+85, 200, 70, 'Credit Ledger', 'Rewards · Debits\nLevel Rates', '#2a2a1a', YELLOW),
        (260, l3_y2+85, 200, 70, 'Evolution Engine', 'Observe → Reflect\nEvolve → Verify', '#1a2a3a', CYAN),
        (480, l3_y2+85, 200, 70, 'Model Adapter', 'RWKV · Transformer\nvLLM · Shimmy', '#3a1a1a', RED),
    ]
    
    for cx, cy, cw, ch, ct, csub, cc_bg, cc_border in core_components:
        draw_rounded_rect(canvas, cx, cy, cx+cw, cy+ch, r=6, fill=cc_bg, outline=cc_border, width=1)
        canvas.create_text(cx+cw//2, cy+18, text=ct, fill=TEXT, font=('Segoe UI', 10, 'bold'), anchor='c')
        for li, line in enumerate(csub.split('\n')):
            canvas.create_text(cx+cw//2, cy+38+li*14, text=line, fill=MUTED, font=('Segoe UI', 8), anchor='c')
    
    # Arrow down
    draw_arrow(canvas, mid_x, l3_y+l3_h, mid_x, l3_y+l3_h+20, color=GREEN)
    
    # === LAYER 4: MODELS ===
    l4_y = l3_y + l3_h + 20
    l4_x, l4_y2 = draw_layer(canvas, 20, l4_y, W-40, 80, 'INFERENCE MODELS', CYAN, [])
    
    models = [
        (60, l4_y2+8, 200, 60, 'RWKV-4-169M', 'CPU · 646MB · 5.8 tok/s\nL0/L1 采集者', CYAN),
        (280, l4_y2+8, 200, 60, 'Qwen2.5-Coder-7B', 'GPU INT4 · L2 标准推理\nL3 重度推理', ACCENT),
        (500, l4_y2+8, 200, 60, 'RWKV-World-430M', 'RAM · L1 轻量推理\n待硬件', PURPLE),
        (720, l4_y2+8, 200, 60, 'Expert LoRA', '微调矩阵\n自进化产物', YELLOW),
    ]
    
    for mx, my, mw, mh, mt, msub, mc in models:
        draw_rounded_rect(canvas, mx, my, mx+mw, my+mh, r=6, fill=PANEL_BG, outline=mc, width=1)
        canvas.create_text(mx+mw//2, my+16, text=mt, fill=TEXT, font=('Segoe UI', 10, 'bold'), anchor='c')
        for li, line in enumerate(msub.split('\n')):
            canvas.create_text(mx+mw//2, my+34+li*12, text=line, fill=MUTED, font=('Segoe UI', 8), anchor='c')
    
    # === RIGHT SIDEBAR: EVOLUTION + CONTRACTS ===
    sidebar_x = W - 270
    
    # Evolution engine detail
    evo_y = l1_y
    draw_rounded_rect(canvas, sidebar_x, evo_y, W-20, evo_y+155, r=8, fill=PANEL_BG, outline=CYAN, width=1.5)
    canvas.create_text(sidebar_x+125, evo_y+18, text='🧬 Evolution Engine', fill=CYAN, font=('Segoe UI', 11, 'bold'), anchor='c')
    
    steps = [
        (sidebar_x+10, evo_y+38, 'Observe', '记录成功/失败', GREEN),
        (sidebar_x+10, evo_y+65, 'Reflect', '分析失败模式', YELLOW),
        (sidebar_x+10, evo_y+92, 'Evolve', 'LoRA 微调矩阵', ACCENT),
        (sidebar_x+10, evo_y+119, 'Verify', '基准测试对比', PURPLE),
    ]
    for sx, sy, st, ssub, sc in steps:
        draw_rounded_rect(canvas, sx, sy, sx+240, sy+24, r=4, fill=BG, outline=sc, width=1)
        canvas.create_text(sx+70, sy+12, text=st, fill=sc, font=('Segoe UI', 9, 'bold'), anchor='c')
        canvas.create_text(sx+165, sy+12, text=ssub, fill=MUTED, font=('Segoe UI', 8), anchor='c')
        if st != 'Verify':
            canvas.create_line(sx+10, sy+24, sx+10, sy+32, fill=BORDER, width=1)
    
    # Smart contracts
    sc_y = evo_y + 165
    draw_rounded_rect(canvas, sidebar_x, sc_y, W-20, sc_y+130, r=8, fill=PANEL_BG, outline=YELLOW, width=1.5)
    canvas.create_text(sidebar_x+125, sc_y+18, text='⛓ Smart Contracts', fill=YELLOW, font=('Segoe UI', 11, 'bold'), anchor='c')
    
    contracts = [
        (sidebar_x+10, sc_y+38, 'D.credits', 'ERC20 · Staking · LevelRates', GREEN),
        (sidebar_x+10, sc_y+65, 'D.reputation', '信誉评分 · Slash机制', PURPLE),
        (sidebar_x+10, sc_y+92, 'D.settlement', '推理结算 · Dispute窗口', BLUE),
        (sidebar_x+10, sc_y+119, 'D.governance', '提案 · 投票 · 执行', YELLOW),
    ]
    for cx, cy, ct, csub, cc in contracts:
        canvas.create_text(cx+8, cy+10, text='●', fill=cc, font=('Segoe UI', 8), anchor='w')
        canvas.create_text(cx+20, cy+10, text=ct, fill=TEXT, font=('Segoe UI', 9, 'bold'), anchor='w')
        canvas.create_text(cx+20, cy+24, text=csub, fill=MUTED, font=('Segoe UI', 8), anchor='w')
    
    # === BOTTOM: Five Level Legend ===
    leg_y = l4_y + 90
    canvas.create_text(60, leg_y+15, text='Five-Level Node System', fill=MUTED, font=('Segoe UI', 10, 'bold'), anchor='w')
    
    levels = [
        ('L0 Collector', 'CPU · ¥2/mo', '#64748b'),
        ('L1 Light', '0.5-1.5B · ¥15/mo', GREEN),
        ('L2 Standard', '7B Q4 · ¥65/mo', BLUE),
        ('L3 Heavy', '14B+ · ¥200/mo', PURPLE),
        ('L4 DataCenter', 'A100 · ¥2000/mo', YELLOW),
    ]
    
    for li, (lt, ls, lc) in enumerate(levels):
        lx = 60 + li * 255
        ly = leg_y + 28
        draw_rounded_rect(canvas, lx, ly, lx+240, ly+45, r=6, fill=PANEL_BG, outline=lc, width=1)
        canvas.create_text(lx+8, ly+14, text='●', fill=lc, font=('Segoe UI', 8), anchor='w')
        canvas.create_text(lx+20, ly+14, text=lt, fill=TEXT, font=('Segoe UI', 10, 'bold'), anchor='w')
        canvas.create_text(lx+20, ly+32, text=ls, fill=MUTED, font=('Segoe UI', 8), anchor='w')
    
    # Footer
    canvas.create_text(W//2, H-10, text='DecentralAI v0.3.0  ·  github.com  ·  MIT License', 
                       fill=MUTED, font=('Segoe UI', 9), anchor='c')
    
    root.update()
    
    # Export to PNG
    if PIL_AVAILABLE:
        ps_file = os.path.join(os.path.dirname(__file__), 'dashboard', 'architecture.ps')
        png_file = os.path.join(os.path.dirname(__file__), 'dashboard', 'architecture.png')
        
        canvas.postscript(file=ps_file, colormode='color')
        img = Image.open(ps_file)
        img.save(png_file, 'png', dpi=(150, 150))
        print(f"Saved: {png_file}")
        
        # Also save deployment diagram
        create_deployment_diagram()
    
    root.mainloop()


def create_deployment_diagram():
    """Create deployment topology diagram"""
    W2, H2 = 1400, 700
    
    root = tk.Tk()
    root.title('DecentralAI Deployment')
    canvas = Canvas(root, width=W2, height=H2, bg=BG, highlightthickness=0)
    canvas.pack()
    
    canvas.create_text(W2//2, 35, text='DecentralAI', fill=TEXT, font=('Segoe UI', 24, 'bold'), anchor='c')
    canvas.create_text(W2//2, 60, text='部署拓扑  |  Deployment Topology', fill=MUTED, font=('Segoe UI', 11), anchor='c')
    
    # === LEFT: Single Node Setup ===
    canvas.create_text(220, 100, text='单节点部署 · Single Node', fill=TEXT, font=('Segoe UI', 14, 'bold'), anchor='c')
    
    # Draw a single node box
    bx, by, bw, bh = 50, 120, 340, 450
    draw_rounded_rect(canvas, bx, by, bx+bw, by+bh, r=10, fill=PANEL_BG, outline=BORDER, width=1)
    
    # Components inside
    components = [
        (bx+20, by+20, 300, 55, '🐧 OS', 'Ubuntu 22.04 / Windows / Raspberry Pi OS', BLUE),
        (bx+20, by+85, 300, 55, '🐍 Python 3.12', '核心运行时 + pip + venv', GREEN),
        (bx+20, by+150, 300, 55, '🔧 DecentralAI', 'run.py / api_server.py / ws_transport.py', ACCENT),
        (bx+20, by+215, 300, 55, '🤖 Model', 'RWKV-4-169M (CPU) / Qwen2.5-Coder (GPU)', PURPLE),
        (bx+20, by+280, 300, 55, '🌐 WebSocket', 'P2P 端口 8001 · API 端口 8000', CYAN),
        (bx+20, by+345, 300, 55, '📊 Dashboard', 'http://localhost:8000/ · 实时监控', YELLOW),
    ]
    for cx, cy, cw, ch, ct, csub, cc in components:
        draw_rounded_rect(canvas, cx, cy, cx+cw, cy+ch, r=6, fill=BG, outline=cc, width=1)
        canvas.create_text(cx+cw//2, cy+16, text=ct, fill=cc, font=('Segoe UI', 10, 'bold'), anchor='c')
        canvas.create_text(cx+cw//2, cy+34, text=csub, fill=MUTED, font=('Segoe UI', 8), anchor='c')
    
    # Quick start commands
    canvas.create_text(bx+20, by+420, text='快速启动', fill=TEXT, font=('Segoe UI', 10, 'bold'), anchor='w')
    commands = [
        'pip install websockets pyyaml torch --target D:\\pylib',
        'git clone https://github.com/.../decentral-ai',
        'cp config.example.yaml config.yaml',
        'python run.py --level L1',
        '# 打开 http://localhost:8000/',
    ]
    for ci, cmd in enumerate(commands):
        canvas.create_text(bx+20, by+445+ci*16, text=f'$ {cmd}', fill=GREEN, font=('Consolas', 8), anchor='w')
    
    # === CENTER: Arrow ===
    ax, ay = bx+bw+20, by+bh//2
    canvas.create_line(ax, ay, ax+60, ay, fill=MUTED, width=2, arrow='last')
    canvas.create_text(ax+30, ay-15, text='or', fill=MUTED, font=('Segoe UI', 10), anchor='c')
    canvas.create_line(ax+60, ay, ax+60, ay-100, fill=MUTED, width=2)
    canvas.create_line(ax+60, ay-100, ax+120, ay-100, fill=MUTED, width=2, arrow='last')
    canvas.create_text(ax+90, ay-115, text='多节点', fill=MUTED, font=('Segoe UI', 9), anchor='c')
    
    # === RIGHT: Multi-Node Cluster ===
    canvas.create_text(900, 100, text='多节点集群 · Multi-Node', fill=TEXT, font=('Segoe UI', 14, 'bold'), anchor='c')
    
    # Cloud boundary
    draw_rounded_rect(canvas, 490, 115, W2-20, H2-20, r=16, fill='#0d1117', outline=BORDER, width=1, dash=(6,4))
    canvas.create_text(755, 140, text='🌐 Internet / LAN', fill=MUTED, font=('Segoe UI', 9), anchor='c')
    
    # Nodes in cluster
    node_positions = [
        (530, 160, 'node-1', 'L2', GREEN, '7B Q4 · GPU'),
        (720, 160, 'node-2', 'L1', BLUE, '1.5B · CPU'),
        (910, 160, 'node-3', 'L2', PURPLE, '7B Q4 · GPU'),
        (530, 300, 'node-4', 'L0', MUTED, '采集者 · CPU'),
        (720, 300, 'node-5', 'L1', YELLOW, '0.5B · CPU'),
        (910, 300, 'node-6', 'L3', ACCENT, '14B+ · 3090'),
    ]
    
    for nx, ny, nm, nl, nc, nsub in node_positions:
        nw, nh = 160, 120
        draw_rounded_rect(canvas, nx, ny, nx+nw, ny+nh, r=8, fill=PANEL_BG, outline=nc, width=1.5)
        canvas.create_text(nx+nw//2, ny+25, text=nm, fill=TEXT, font=('Segoe UI', 11, 'bold'), anchor='c')
        draw_rounded_rect(canvas, nx+nw//2-25, ny+40, nx+nw//2+25, ny+65, r=4, fill=nc, outline='')
        canvas.create_text(nx+nw//2, ny+53, text=nl, fill='white', font=('Segoe UI', 14, 'bold'), anchor='c')
        canvas.create_text(nx+nw//2, ny+82, text=nsub, fill=MUTED, font=('Segoe UI', 8), anchor='c')
        canvas.create_text(nx+nw//2, ny+100, text=f'API:{(nx-500)*100+8000}', fill=MUTED, font=('Segoe UI', 8), anchor='c')
    
    # Blockchain box
    bc_y = 450
    draw_rounded_rect(canvas, 530, bc_y, W2-30, bc_y+130, r=8, fill='#1a1520', outline=YELLOW, width=1.5)
    canvas.create_text(720, bc_y+18, text='⛓ Blockchain Layer (可选)', fill=YELLOW, font=('Segoe UI', 11, 'bold'), anchor='c')
    
    chains = [
        (550, bc_y+40, 'Scroll', 'L2 · 主推荐', '#6366f1'),
        (700, bc_y+40, 'zkSync', 'L2 · zkEVM', '#8b5cf6'),
        (850, bc_y+40, '长安链', '联盟链 · 国内', '#22c55e'),
        (550, bc_y+80, 'Arbitrum', 'L2 · 生态最大', '#28a0f0'),
        (700, bc_y+80, 'Polygon', 'L2 · 低费用', '#8247e5'),
        (850, bc_y+80, 'Local', 'Hardhat · 开发', '#eab308'),
    ]
    for cx, cy, ct, csub, cc in chains:
        draw_rounded_rect(canvas, cx, cy, cx+130, cy+32, r=4, fill=BG, outline=cc, width=1)
        canvas.create_text(cx+65, cy+11, text=ct, fill=cc, font=('Segoe UI', 9, 'bold'), anchor='c')
        canvas.create_text(cx+65, cy+24, text=csub, fill=MUTED, font=('Segoe UI', 8), anchor='c')
    
    # Connection to blockchain
    for nx, ny in [(600, 300), (790, 300), (980, 300)]:
        canvas.create_line(nx+80, ny+120, nx+80, bc_y, fill=YELLOW, dash=(3,3), width=1)
        canvas.create_oval(nx+80-3, bc_y-3, nx+80+3, bc_y+3, fill=YELLOW)
    
    root.update()
    
    if PIL_AVAILABLE:
        ps_file2 = os.path.join(os.path.dirname(__file__), 'dashboard', 'deployment.ps')
        png_file2 = os.path.join(os.path.dirname(__file__), 'dashboard', 'deployment.png')
        canvas.postscript(file=ps_file2, colormode='color')
        img2 = Image.open(ps_file2)
        img2.save(png_file2, 'png', dpi=(150, 150))
        print(f"Saved: {png_file2}")
    
    root.mainloop()


if __name__ == "__main__":
    create_diagram()
