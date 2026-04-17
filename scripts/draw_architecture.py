"""
DecentralAI Architecture Diagram - Pure PIL
=============================================
No tkinter needed, saves directly to PNG.
"""

import sys
import os
sys.path.insert(0, r'D:\pylib')

from PIL import Image, ImageDraw, ImageFont

# ============================================================
# Colors
# ============================================================
BG      = '#0d1117'
PANEL   = '#161b22'
BORDER  = '#30363d'
TEXT    = '#e6edf3'
MUTED   = '#8b949e'
ACCENT  = '#6366f1'
GREEN   = '#22c55e'
BLUE    = '#3b82f6'
PURPLE  = '#a855f7'
YELLOW  = '#eab308'
CYAN    = '#06b6d4'
RED     = '#ef4444'
ORANGE  = '#f97316'

W, H = 1400, 920


def load_font(size, bold=False):
    """Load a font, trying multiple sources"""
    weight = 'bold' if bold else 'normal'
    
    # Try system fonts first (fast)
    font_paths = [
        r'C:\Windows\Fonts\segoeui.ttf',
        r'C:\Windows\Fonts\seguisb.ttf',
        r'C:\Windows\Fonts\msyh.ttc',
        r'C:\Windows\Fonts\simhei.ttf',
        r'C:\Windows\Fonts\cour.ttf',
        r'C:\Windows\Fonts\arial.ttf',
    ]
    
    for fp in font_paths:
        try:
            return ImageFont.truetype(fp, size, encoding='utf-8')
        except:
            pass
    
    # Fallback to default
    try:
        return ImageFont.truetype('arial.ttf', size)
    except:
        return ImageFont.load_default()


def rrect(draw, xy, radius=8, fill=None, outline=None, width=1):
    """Draw a rounded rectangle"""
    x1, y1, x2, y2 = xy
    if fill:
        draw.rectangle([x1+radius, y1, x2-radius, y2], fill=fill)
        draw.rectangle([x1, y1+radius, x2, y2-radius], fill=fill)
        draw.ellipse([x1, y1, x1+2*radius, y1+2*radius], fill=fill)
        draw.ellipse([x2-2*radius, y1, x2, y1+2*radius], fill=fill)
        draw.ellipse([x1, y2-2*radius, x1+2*radius, y2], fill=fill)
        draw.ellipse([x2-2*radius, y2-2*radius, x2, y2], fill=fill)
    if outline:
        # Top
        draw.line([x1+radius, y1, x2-radius, y1], fill=outline, width=width)
        draw.line([x1, y1+radius, x1, y2-radius], fill=outline, width=width)
        draw.line([x1+radius, y2, x2-radius, y2], fill=outline, width=width)
        draw.line([x2, y1+radius, x2, y2-radius], fill=outline, width=width)
        # Corners
        draw.arc([x1, y1, x1+2*radius, y1+2*radius], 180, 270, fill=outline, width=width)
        draw.arc([x2-2*radius, y1, x2, y1+2*radius], 270, 360, fill=outline, width=width)
        draw.arc([x1, y2-2*radius, x1+2*radius, y2], 90, 180, fill=outline, width=width)
        draw.arc([x2-2*radius, y2-2*radius, x2, y2], 0, 90, fill=outline, width=width)


def draw_panel(draw, x, y, w, h, title, subtitle='', color=ACCENT, radius=8):
    """Draw a panel/card with title"""
    rrect(draw, [x, y, x+w, y+h], radius=radius, fill=PANEL, outline=color, width=1)
    
    if title:
        tfont = load_font(11, bold=True)
        draw.text((x + w//2, y + 18), title, fill=TEXT, font=tfont, anchor='mm')
    if subtitle:
        sfont = load_font(9)
        draw.text((x + w//2, y + 36), subtitle, fill=MUTED, font=sfont, anchor='mm')


def draw_arrow_v(draw, x, y1, y2, color=MUTED):
    """Vertical arrow"""
    draw.line([x, y1, x, y2-6], fill=color, width=1)
    draw.polygon([(x, y2), (x-4, y2-8), (x+4, y2-8)], fill=color)


def draw_arrow_h(draw, x1, y, x2, color=MUTED):
    """Horizontal arrow"""
    draw.line([x1, y, x2-6, y], fill=color, width=1)
    draw.polygon([(x2, y), (x2-8, y-4), (x2-8, y+4)], fill=color)


def draw_layer_band(draw, x, y, w, h, label, color, font_size=10):
    """Draw a horizontal layer band with label"""
    rrect(draw, [x, y, x+w, y+h], radius=10, fill=BG, outline=BORDER, width=1)
    lfont = load_font(font_size, bold=True)
    lW = len(label) * font_size * 0.65 + 20
    rrect(draw, [x+6, y+4, x+6+int(lW), y+h-4], radius=5, fill=color)
    draw.text((x+6+int(lW)//2, y+h//2), label, fill='white', font=lfont, anchor='mm')


def main():
    img = Image.new('RGB', (W, H), BG)
    draw = ImageDraw.Draw(img)
    
    ftitle = load_font(22, bold=True)
    fsubtitle = load_font(10)
    fbold = load_font(11, bold=True)
    fsmall = load_font(9)
    fmono = load_font(8)
    
    # ============================================================
    # TITLE
    # ============================================================
    draw.text((W//2, 28), 'DecentralAI', fill=TEXT, font=ftitle, anchor='mm')
    draw.text((W//2, 56), '去中心化异构 AI 推理网络  |  dMoE Architecture  |  v0.3.0', 
              fill=MUTED, font=fsubtitle, anchor='mm')
    
    # ============================================================
    # LAYER 0: CLIENT
    # ============================================================
    L0_Y = 72
    draw_layer_band(draw, 20, L0_Y, W-40, 72, 'CLIENT LAYER', BLUE)
    
    apps = [
        ('OpenAI SDK', 'openai-python'),
        ('REST API', 'curl / HTTP'),
        ('Web UI', 'dashboard'),
        ('CLI Tool', 'evoagent'),
    ]
    for i, (name, sub) in enumerate(apps):
        ax = 50 + i * 325
        rrect(draw, [ax, L0_Y+28, ax+285, L0_Y+58], radius=6, fill=PANEL, outline=BLUE, width=1)
        draw.text((ax+142, L0_Y+36), name, fill=TEXT, font=fbold, anchor='mm')
        draw.text((ax+142, L0_Y+50), sub, fill=MUTED, font=fsmall, anchor='mm')
    
    draw_arrow_v(draw, W//2, L0_Y+72, L0_Y+92, BLUE)
    
    # ============================================================
    # LAYER 1: API & GATEWAY
    # ============================================================
    L1_Y = L0_Y + 92
    draw_layer_band(draw, 20, L1_Y, W-40, 90, 'API & GATEWAY', PURPLE)
    
    # Gateway panel
    gx, gy, gw, gh = W//2-300, L1_Y+28, 600, 52
    rrect(draw, [gx, gy, gx+gw, gy+gh], radius=8, fill=PANEL, outline=PURPLE, width=1)
    draw.text((gx+gw//2, gy+18), 'API Gateway (api_server.py)', fill=PURPLE, font=fbold, anchor='mm')
    eps = '  ·  '.join(['/v1/chat/completions', '/v1/completions', '/v1/models', '/health', '/status', '/dashboard'])
    draw.text((gx+gw//2, gy+38), eps, fill=MUTED, font=fsmall, anchor='mm')
    
    draw_arrow_v(draw, W//2, L1_Y+90, L1_Y+110, PURPLE)
    
    # ============================================================
    # LAYER 2: P2P NETWORK
    # ============================================================
    L2_Y = L1_Y + 110
    draw_layer_band(draw, 20, L2_Y, W-40, 100, 'P2P NETWORK LAYER (ws_transport.py)', ACCENT)
    
    # Five nodes
    node_data = [
        ('node_alpha', 'L2 · Standard', '#22c55e'),
        ('node_beta', 'L1 · Light', BLUE),
        ('Bootstrap', 'L4 · DataCenter', YELLOW),
        ('node_gamma', 'L2 · Standard', ACCENT),
        ('node_delta', 'L3 · Heavy', PURPLE),
    ]
    
    node_xs = [W//2 - 550, W//2 - 300, W//2, W//2 + 300, W//2 + 550]
    
    # Connection lines
    for i in range(len(node_xs)-1):
        x1, x2 = node_xs[i]+90, node_xs[i+1]-90
        y = L2_Y+55
        draw.line([x1, y, x2, y], fill=BORDER, width=1)
        draw.ellipse([x1-3, y-3, x1+3, y+3], fill=ACCENT)
        draw.ellipse([x2-3, y-3, x2+3, y+3], fill=ACCENT)
    
    for i, ((nx, nl, nc), x) in enumerate(zip(node_data, node_xs)):
        rrect(draw, [x, L2_Y+22, x+180, L2_Y+90], radius=6, fill=PANEL, outline=nc, width=1)
        draw.text((x+90, L2_Y+38), nx, fill=TEXT, font=fbold, anchor='mm')
        draw.text((x+90, L2_Y+52), nl, fill=MUTED, font=fsmall, anchor='mm')
        draw.text((x+90, L2_Y+66), 'WebSocket P2P', fill=MUTED, font=fsmall, anchor='mm')
    
    draw_arrow_v(draw, W//2, L2_Y+100, L2_Y+120, ACCENT)
    
    # ============================================================
    # LAYER 3: CORE ENGINE (left)
    # ============================================================
    L3_Y = L2_Y + 120
    draw_layer_band(draw, 20, L3_Y, W-310, 190, 'CORE ENGINE (core.py)', GREEN)
    
    core_items = [
        # Row 1
        (30, L3_Y+8, 'Node Identity', 'Capabilities · Experts\nLevel · Reputation', BLUE),
        (255, L3_Y+8, 'Router', 'Expert Selection\nDomain Routing', GREEN),
        (480, L3_Y+8, 'Verifier', 'Consensus\nSyntax Check', PURPLE),
        # Row 2
        (30, L3_Y+100, 'Credit Ledger', 'Rewards · Debits\nLevel Rates', YELLOW),
        (255, L3_Y+100, 'Evolution Engine', 'Observe → Reflect\nEvolve → Verify', CYAN),
        (480, L3_Y+100, 'Model Adapter', 'RWKV · Transformer\nvLLM · Shimmy', RED),
    ]
    
    for cx, cy, ct, csub, cc in core_items:
        rrect(draw, [cx, cy, cx+200, cy+82], radius=6, fill=BG, outline=cc, width=1)
        draw.text((cx+100, cy+16), ct, fill=cc, font=fbold, anchor='mm')
        lines = csub.split('\n')
        for li, line in enumerate(lines):
            draw.text((cx+100, cy+36+li*16), line, fill=MUTED, font=fsmall, anchor='mm')
    
    # ============================================================
    # RIGHT SIDEBAR: Evolution + Contracts
    # ============================================================
    SX = W - 290
    
    # Evolution Engine
    EY = L1_Y
    rrect(draw, [SX, EY, W-20, EY+155], radius=8, fill=PANEL, outline=CYAN, width=1)
    draw.text((SX+135, EY+18), 'Evolution Engine', fill=CYAN, font=fbold, anchor='mm')
    
    evo_steps = [
        ('Observe', '记录成功/失败', GREEN),
        ('Reflect', '分析失败模式', YELLOW),
        ('Evolve', 'LoRA 微调矩阵', ACCENT),
        ('Verify', '基准测试对比', PURPLE),
    ]
    for i, (st, ssub, sc) in enumerate(evo_steps):
        sy = EY + 40 + i * 27
        rrect(draw, [SX+10, sy, SX+260, sy+22], radius=4, fill=BG, outline=sc, width=1)
        draw.text((SX+75, sy+11), st, fill=sc, font=load_font(9, bold=True), anchor='mm')
        draw.text((SX+170, sy+11), ssub, fill=MUTED, font=fsmall, anchor='mm')
        if i < 3:
            draw.line([SX+135, sy+22, SX+135, sy+27], fill=BORDER, width=1)
    
    # Smart Contracts
    CY2 = EY + 165
    rrect(draw, [SX, CY2, W-20, CY2+135], radius=8, fill=PANEL, outline=YELLOW, width=1)
    draw.text((SX+135, CY2+18), 'Smart Contracts', fill=YELLOW, font=fbold, anchor='mm')
    
    contracts = [
        ('D.Credits', 'ERC20 · Staking · LevelRates', GREEN),
        ('D.Reputation', '信誉评分 · Slash机制', PURPLE),
        ('D.Settlement', '推理结算 · Dispute窗口', BLUE),
        ('D.Governance', '提案 · 投票 · 执行', YELLOW),
    ]
    for i, (ct, csub, cc) in enumerate(contracts):
        cy = CY2 + 40 + i * 24
        draw.ellipse([SX+18, cy+3, SX+26, cy+11], fill=cc)
        draw.text((SX+30, cy+7), ct, fill=TEXT, font=load_font(9, bold=True))
        draw.text((SX+30, cy+19), csub, fill=MUTED, font=fsmall)
    
    # ============================================================
    # LAYER 4: MODELS
    # ============================================================
    L4_Y = L3_Y + 190
    draw_layer_band(draw, 20, L4_Y, W-310, 80, 'INFERENCE MODELS', CYAN)
    
    models = [
        ('RWKV-4-169M', 'CPU · 646MB\n5.8 tok/s · L0/L1', CYAN),
        ('Qwen2.5-Coder-7B', 'GPU INT4 · L2\n目标模型', ACCENT),
        ('RWKV-World-430M', 'RAM · L1\n待硬件', PURPLE),
        ('Expert LoRA', '微调矩阵\n自进化产物', YELLOW),
    ]
    
    for i, (mt, msub, mc) in enumerate(models):
        mx = 30 + i * 305
        rrect(draw, [mx, L4_Y+8, mx+280, L4_Y+62], radius=6, fill=PANEL, outline=mc, width=1)
        draw.text((mx+140, L4_Y+22), mt, fill=TEXT, font=fbold, anchor='mm')
        lines = msub.split('\n')
        for li, line in enumerate(lines):
            draw.text((mx+140, L4_Y+38+li*12), line, fill=MUTED, font=fsmall, anchor='mm')
    
    # ============================================================
    # BOTTOM: Five-Level Legend
    # ============================================================
    LY = L4_Y + 90
    
    draw.text((50, LY+8), 'Five-Level Node System', fill=MUTED, font=fbold)
    
    levels = [
        ('L0 Collector', 'CPU · ¥2/月', '#64748b'),
        ('L1 Light', '0.5-1.5B · ¥15/月', GREEN),
        ('L2 Standard', '7B Q4 · ¥65/月', BLUE),
        ('L3 Heavy', '14B+ · ¥200/月', PURPLE),
        ('L4 DataCenter', 'A100 · ¥2000/月', YELLOW),
    ]
    
    for i, (lt, ls, lc) in enumerate(levels):
        lx = 50 + i * 258
        rrect(draw, [lx, LY+28, lx+245, LY+78], radius=6, fill=PANEL, outline=lc, width=1)
        draw.ellipse([lx+10, LY+44, lx+18, LY+52], fill=lc)
        draw.text((lx+24, LY+48), lt, fill=TEXT, font=fbold)
        draw.text((lx+24, LY+64), ls, fill=MUTED, font=fsmall)
    
    # Footer
    draw.text((W//2, H-12), 'github.com  ·  MIT License  ·  v0.3.0', fill=MUTED, font=fsmall, anchor='mm')
    
    # ============================================================
    # SAVE
    # ============================================================
    out_dir = os.path.dirname(__file__)
    out_path = os.path.join(out_dir, '..', 'dashboard', 'architecture.png')
    out_path = os.path.abspath(out_path)
    
    # Crop empty bottom
    img.save(out_path, 'PNG', optimize=True)
    print(f'Saved: {out_path}')
    print(f'Size: {img.size}')
    
    return out_path


if __name__ == '__main__':
    main()
