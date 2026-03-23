/**
 * Pixel Agents Trading Floor — Canvas 2D Game Engine
 * Retro RPG office style with warm pixel art aesthetic.
 */

// ── Warm Color Palette ──────────────────────────────────────────────────
const C = {
    // Walls & background
    wallDark: '#1a2332', wallMid: '#243447', wallLight: '#2d4156',
    baseboard: '#0f1820',
    // Floor (warm wood)
    floorA: '#8B7355', floorB: '#7A6548', floorGrain: '#6e5a3e',
    floorHighlight: '#9a8264',
    // Desk
    deskFront: '#6B4F2E', deskTop: '#C4A265', deskSide: '#5a4025',
    deskEdge: '#8B6914', deskHandle: '#3d2e1a',
    // Monitor
    monFrame: '#2a2a2a', monScreen: '#0a1520', monStand: '#3a3a3a',
    // Furniture
    chairSeat: '#b89868', chairBack: '#a08050', chairLeg: '#6e5530',
    shelfWood: '#7a5c30', shelfDark: '#5a4020',
    bookColors: ['#c94040', '#4080c0', '#40a060', '#c0a030', '#a050a0', '#50b0b0'],
    // Character
    skin: '#e8c4a0', skinShadow: '#d4a574', skinLight: '#f0d6b4',
    hairDark: '#2d333b', hairBrown: '#6e4b00', hairBlack: '#1c1c1c', hairWavy: '#4a3728',
    eyeWhite: '#f0f0f0', eyePupil: '#1a1a2e',
    pantsDark: '#2a3040', shoes: '#1a1a1a',
    // Decor
    plantGreen: '#3a8a3a', plantDark: '#2a6a2a', potBrown: '#8a6830',
    mugWhite: '#d0d0d0', mugHandle: '#a0a0a0',
    // Agent shirt colors
    shirtBlue: '#4a90d9', shirtBlueDark: '#3570b0',
    shirtOrange: '#d4952a', shirtOrangeDark: '#a87520',
    shirtRed: '#d94444', shirtRedDark: '#b03030',
    shirtGreen: '#45a54a', shirtGreenDark: '#308035',
    // UI
    text: '#c9d1d9', textDim: '#8b949e',
    blue: '#58a6ff', green: '#3fb950', red: '#f85149', yellow: '#d29922',
};

const AGENT_STYLES = {
    market: { shirt: C.shirtBlue, shirtDk: C.shirtBlueDark, hair: C.hairDark, hairStyle: 'neat', accessory: 'glasses', monGlow: '#4a90d9' },
    data:   { shirt: C.shirtBlue, shirtDk: C.shirtBlueDark, hair: C.hairDark, hairStyle: 'neat', accessory: 'glasses', monGlow: '#4a90d9' },
    news:   { shirt: C.shirtOrange, shirtDk: C.shirtOrangeDark, hair: C.hairBrown, hairStyle: 'messy', accessory: 'badge', monGlow: '#d4952a' },
    signal: { shirt: C.shirtRed, shirtDk: C.shirtRedDark, hair: C.hairBlack, hairStyle: 'spiky', accessory: 'headphones', monGlow: '#d94444' },
    global: { shirt: C.shirtGreen, shirtDk: C.shirtGreenDark, hair: C.hairWavy, hairStyle: 'wavy', accessory: 'none', monGlow: '#45a54a' },
};

const STATE_ANIMATION = {
    'running': 'typing', 'completed': 'idle', 'failed': 'alert',
    'waiting': 'looking', 'skipped': 'sleeping', 'scheduled': 'reading', 'idle': 'idle',
};

// ── Helper: draw a filled pixel rect ─────────────────────────────────
function px(ctx, x, y, w, h, color) {
    ctx.fillStyle = color;
    ctx.fillRect(x, y, w, h);
}

// ── Character Class ──────────────────────────────────────────────────
class AgentCharacter {
    constructor(name, role, x, y) {
        this.name = name;
        this.role = role;
        this.x = x;
        this.y = y;
        this.style = AGENT_STYLES[role.toLowerCase()] || AGENT_STYLES.data;
        this.state = 'idle';
        this.animation = 'idle';
        this.progress = '';
        this.frame = 0;
        this.frameTimer = 0;
        this.bobOffset = 0;
        this.speechBubble = '';
        this.speechTimer = 0;
        this.hitbox = { x: 0, y: 0, w: 0, h: 0 };
        this.runHitbox = { x: 0, y: 0, w: 0, h: 0 };
    }

    setState(state, progress) {
        const changed = this.state !== state || this.progress !== (progress || '');
        this.state = state;
        this.animation = STATE_ANIMATION[state] || 'idle';
        this.progress = progress || '';
        // Only show speech bubble when state/progress actually changes
        if (changed && progress) {
            this.speechBubble = progress;
            this.speechTimer = 300;
        }
    }

    update() {
        this.frameTimer++;
        if (this.frameTimer % 10 === 0) this.frame = (this.frame + 1) % 4;

        if (this.animation === 'typing') {
            this.bobOffset = Math.sin(this.frameTimer * 0.12) * 1;
        } else if (this.animation === 'sleeping') {
            this.bobOffset = Math.sin(this.frameTimer * 0.04) * 0.5 + 2;
        } else if (this.animation === 'alert') {
            this.bobOffset = Math.sin(this.frameTimer * 0.5) * 1;
        } else {
            this.bobOffset = Math.sin(this.frameTimer * 0.03) * 0.3;
        }
        if (this.speechTimer > 0) this.speechTimer--;
    }

    draw(ctx, s) {
        const cx = this.x * s;
        const cy = (this.y + this.bobOffset) * s;
        const p = s; // pixel unit
        const st = this.style;

        ctx.save();

        // ── Shadow on floor ──
        ctx.fillStyle = 'rgba(0,0,0,0.2)';
        ctx.beginPath();
        ctx.ellipse(cx, cy + 2 * p, 7 * p, 2 * p, 0, 0, Math.PI * 2);
        ctx.fill();

        // ── Shoes ──
        px(ctx, cx - 3 * p, cy - 1 * p, 2.5 * p, 2 * p, C.shoes);
        px(ctx, cx + 0.5 * p, cy - 1 * p, 2.5 * p, 2 * p, C.shoes);

        // ── Pants ──
        px(ctx, cx - 3 * p, cy - 5 * p, 2.5 * p, 4 * p, C.pantsDark);
        px(ctx, cx + 0.5 * p, cy - 5 * p, 2.5 * p, 4 * p, C.pantsDark);
        // Belt line
        px(ctx, cx - 4 * p, cy - 5 * p, 8 * p, 1 * p, '#1e2530');

        // ── Shirt/torso ──
        px(ctx, cx - 4 * p, cy - 13 * p, 8 * p, 8 * p, st.shirt);
        // Shirt shadow/fold
        px(ctx, cx - 4 * p, cy - 13 * p, 1 * p, 8 * p, st.shirtDk);
        px(ctx, cx + 3 * p, cy - 13 * p, 1 * p, 8 * p, st.shirtDk);
        // Collar
        px(ctx, cx - 2 * p, cy - 13 * p, 1.5 * p, 1.5 * p, '#e8e8e8');
        px(ctx, cx + 0.5 * p, cy - 13 * p, 1.5 * p, 1.5 * p, '#e8e8e8');

        // ── Arms ──
        const armY = cy - 11 * p;
        if (this.animation === 'typing') {
            const off = (this.frame % 2) * p;
            // Left arm - forearm on desk
            px(ctx, cx - 6 * p, armY + off, 2 * p, 5 * p, st.shirt);
            px(ctx, cx - 6 * p, armY + 4 * p + off, 2 * p, 2 * p, C.skin);
            // Right arm
            px(ctx, cx + 4 * p, armY - off, 2 * p, 5 * p, st.shirt);
            px(ctx, cx + 4 * p, armY + 4 * p - off, 2 * p, 2 * p, C.skin);
        } else if (this.animation === 'sleeping') {
            // Arms on desk
            px(ctx, cx - 6 * p, armY + 2 * p, 2 * p, 4 * p, st.shirt);
            px(ctx, cx - 6 * p, armY + 5 * p, 2 * p, 2 * p, C.skin);
            px(ctx, cx + 4 * p, armY + 2 * p, 2 * p, 4 * p, st.shirt);
            px(ctx, cx + 4 * p, armY + 5 * p, 2 * p, 2 * p, C.skin);
        } else {
            // Arms at rest
            px(ctx, cx - 5.5 * p, armY, 1.5 * p, 5 * p, st.shirt);
            px(ctx, cx - 5.5 * p, armY + 4 * p, 1.5 * p, 2 * p, C.skin);
            px(ctx, cx + 4 * p, armY, 1.5 * p, 5 * p, st.shirt);
            px(ctx, cx + 4 * p, armY + 4 * p, 1.5 * p, 2 * p, C.skin);
        }

        // ── Neck ──
        px(ctx, cx - 1 * p, cy - 14 * p, 2 * p, 2 * p, C.skin);

        // ── Head ──
        const headY = this.animation === 'sleeping' ? cy - 19 * p : cy - 21 * p;
        // Face
        px(ctx, cx - 4 * p, headY, 8 * p, 7 * p, C.skin);
        // Ear shadows
        px(ctx, cx - 4.5 * p, headY + 2 * p, 1 * p, 2 * p, C.skinShadow);
        px(ctx, cx + 3.5 * p, headY + 2 * p, 1 * p, 2 * p, C.skinShadow);

        // ── Hair ──
        const hairColor = st.hair;
        if (st.hairStyle === 'neat') {
            // Short neat hair
            px(ctx, cx - 4.5 * p, headY - 2 * p, 9 * p, 3 * p, hairColor);
            px(ctx, cx - 4.5 * p, headY + 0.5 * p, 1.5 * p, 2 * p, hairColor);
            px(ctx, cx + 3 * p, headY + 0.5 * p, 1.5 * p, 2 * p, hairColor);
        } else if (st.hairStyle === 'messy') {
            // Messy/curly
            px(ctx, cx - 5 * p, headY - 3 * p, 10 * p, 4 * p, hairColor);
            px(ctx, cx - 5 * p, headY + 0.5 * p, 2 * p, 3 * p, hairColor);
            px(ctx, cx + 3 * p, headY + 0.5 * p, 2 * p, 3 * p, hairColor);
            // Curl tufts
            px(ctx, cx - 5 * p, headY - 3 * p, 2 * p, 1 * p, hairColor);
            px(ctx, cx + 4 * p, headY - 3 * p, 2 * p, 1 * p, hairColor);
        } else if (st.hairStyle === 'spiky') {
            // Spiky upward
            px(ctx, cx - 4 * p, headY - 2 * p, 8 * p, 3 * p, hairColor);
            px(ctx, cx - 3 * p, headY - 4 * p, 2 * p, 2 * p, hairColor);
            px(ctx, cx + 0 * p, headY - 5 * p, 2 * p, 3 * p, hairColor);
            px(ctx, cx + 2 * p, headY - 3 * p, 2 * p, 1 * p, hairColor);
        } else if (st.hairStyle === 'wavy') {
            // Longer wavy
            px(ctx, cx - 5 * p, headY - 2 * p, 10 * p, 3 * p, hairColor);
            px(ctx, cx - 5 * p, headY + 0.5 * p, 2 * p, 5 * p, hairColor);
            px(ctx, cx + 3.5 * p, headY + 0.5 * p, 2 * p, 5 * p, hairColor);
        }

        // ── Eyes ──
        if (this.animation === 'sleeping') {
            // Closed eyes - horizontal lines
            px(ctx, cx - 3 * p, headY + 3 * p, 2 * p, 0.7 * p, C.eyePupil);
            px(ctx, cx + 1 * p, headY + 3 * p, 2 * p, 0.7 * p, C.eyePupil);
        } else if (this.animation === 'alert') {
            // Wide alarmed eyes
            px(ctx, cx - 3 * p, headY + 2 * p, 2.5 * p, 2.5 * p, C.eyeWhite);
            px(ctx, cx + 0.5 * p, headY + 2 * p, 2.5 * p, 2.5 * p, C.eyeWhite);
            px(ctx, cx - 2 * p, headY + 2.5 * p, 1.5 * p, 1.5 * p, C.red);
            px(ctx, cx + 1.5 * p, headY + 2.5 * p, 1.5 * p, 1.5 * p, C.red);
        } else {
            // Normal eyes with whites and pupils
            px(ctx, cx - 3 * p, headY + 2.5 * p, 2 * p, 2 * p, C.eyeWhite);
            px(ctx, cx + 1 * p, headY + 2.5 * p, 2 * p, 2 * p, C.eyeWhite);
            // Pupils - shift if looking
            const pupilOff = this.animation === 'looking' ? 0.7 * p : 0;
            px(ctx, cx - 2.5 * p + pupilOff, headY + 3 * p, 1.2 * p, 1.2 * p, C.eyePupil);
            px(ctx, cx + 1.5 * p + pupilOff, headY + 3 * p, 1.2 * p, 1.2 * p, C.eyePupil);

            // Blink
            if (this.frame === 3 && this.frameTimer % 150 < 8) {
                px(ctx, cx - 3 * p, headY + 2.5 * p, 2 * p, 2 * p, C.skin);
                px(ctx, cx + 1 * p, headY + 2.5 * p, 2 * p, 2 * p, C.skin);
                px(ctx, cx - 3 * p, headY + 3 * p, 2 * p, 0.7 * p, C.eyePupil);
                px(ctx, cx + 1 * p, headY + 3 * p, 2 * p, 0.7 * p, C.eyePupil);
            }
        }

        // ── Mouth ──
        px(ctx, cx - 1 * p, headY + 5.5 * p, 2 * p, 0.6 * p, C.skinShadow);

        // ── Accessories ──
        if (st.accessory === 'glasses') {
            ctx.strokeStyle = '#8090a0';
            ctx.lineWidth = 0.7 * p;
            ctx.strokeRect(cx - 3.5 * p, headY + 2 * p, 2.8 * p, 2.5 * p);
            ctx.strokeRect(cx + 0.5 * p, headY + 2 * p, 2.8 * p, 2.5 * p);
            ctx.beginPath();
            ctx.moveTo(cx - 0.7 * p, headY + 3 * p);
            ctx.lineTo(cx + 0.5 * p, headY + 3 * p);
            ctx.stroke();
        } else if (st.accessory === 'badge') {
            // Press badge on chest
            px(ctx, cx - 1 * p, cy - 11 * p, 3 * p, 2 * p, '#e8e8e8');
            px(ctx, cx - 0.5 * p, cy - 10.5 * p, 2 * p, 1 * p, C.red);
        } else if (st.accessory === 'headphones') {
            // Headphone arc
            ctx.strokeStyle = '#555';
            ctx.lineWidth = 1.2 * p;
            ctx.beginPath();
            ctx.arc(cx, headY - 0.5 * p, 5.5 * p, Math.PI, 0);
            ctx.stroke();
            // Ear pads
            px(ctx, cx - 6 * p, headY + 1 * p, 2 * p, 3 * p, '#444');
            px(ctx, cx + 4 * p, headY + 1 * p, 2 * p, 3 * p, '#444');
        }

        // ── Sleeping Z's ──
        if (this.animation === 'sleeping') {
            const zPhase = (this.frameTimer * 0.02) % 1;
            ctx.fillStyle = C.textDim;
            ctx.font = `bold ${5 * p}px monospace`;
            ctx.textAlign = 'center';
            ctx.globalAlpha = 1 - zPhase;
            ctx.fillText('z', cx + 5 * p, headY - 5 * p * zPhase);
            ctx.font = `bold ${4 * p}px monospace`;
            ctx.globalAlpha = 0.6;
            ctx.fillText('z', cx + 8 * p, headY - 10 * p * zPhase);
            ctx.globalAlpha = 1;
        }

        // ── Alert exclamation ──
        if (this.animation === 'alert') {
            const bounce = Math.abs(Math.sin(this.frameTimer * 0.15)) * 3 * p;
            px(ctx, cx - 1 * p, headY - 8 * p - bounce, 2 * p, 4 * p, C.red);
            px(ctx, cx - 1 * p, headY - 3 * p - bounce, 2 * p, 1.5 * p, C.red);
        }

        // ── Status dot ──
        let dotColor = C.textDim;
        if (this.state === 'completed') dotColor = C.green;
        else if (this.state === 'running') dotColor = C.blue;
        else if (this.state === 'failed') dotColor = C.red;
        else if (this.state === 'waiting') dotColor = C.yellow;

        ctx.fillStyle = dotColor;
        ctx.beginPath();
        ctx.arc(cx, headY - 4 * p, 1.5 * p, 0, Math.PI * 2);
        ctx.fill();
        // Glow
        ctx.globalAlpha = 0.3;
        ctx.beginPath();
        ctx.arc(cx, headY - 4 * p, 3 * p, 0, Math.PI * 2);
        ctx.fill();
        ctx.globalAlpha = 1;

        // ── Name label ──
        ctx.fillStyle = '#d0c8b8';
        ctx.font = `bold ${6 * p}px monospace`;
        ctx.textAlign = 'center';
        ctx.fillText(this.role, cx, cy + 7 * p);

        // ── Run / Stop button ──
        const btnW = 20 * p, btnH = 7 * p;
        const btnX = cx - btnW / 2, btnY = cy + 9 * p;
        const isRunning = this.state === 'running';
        ctx.fillStyle = isRunning ? '#3a1a1a' : '#1a3525';
        ctx.strokeStyle = isRunning ? '#f85149' : '#2ea043';
        ctx.lineWidth = 0.8 * p;
        ctx.beginPath();
        ctx.roundRect(btnX, btnY, btnW, btnH, 2 * p);
        ctx.fill();
        ctx.stroke();
        ctx.fillStyle = isRunning ? '#f85149' : '#3fb950';
        ctx.font = `bold ${4.5 * p}px monospace`;
        ctx.textAlign = 'center';
        ctx.fillText(isRunning ? '\u25A0 STOP' : '\u25BA RUN', cx, btnY + 5 * p);
        ctx.font = `${2.5 * p}px monospace`;
        ctx.fillStyle = 'rgba(139,148,158,0.7)';
        ctx.fillText('double-click', cx, btnY + 8.5 * p);
        this.runHitbox = { x: btnX, y: btnY, w: btnW, h: btnH };

        // ── Progress bar (when running) ──
        if (isRunning) {
            const barW = btnW, barH = 2.5 * p;
            const barX = btnX, barY = btnY + btnH + 1.5 * p;
            // Try to parse "N/M" from progress string for determinate fill
            let fillPct = -1;
            const match = this.progress && this.progress.match(/(\d+)[\s\/]+(\d+)/);
            if (match) {
                const done = parseInt(match[1]), total = parseInt(match[2]);
                if (total > 0) fillPct = Math.min(done / total, 1);
            }
            // Background track
            ctx.fillStyle = '#21262d';
            ctx.beginPath();
            ctx.roundRect(barX, barY, barW, barH, 1 * p);
            ctx.fill();
            // Fill — determinate or animated indeterminate
            ctx.fillStyle = '#58a6ff';
            ctx.beginPath();
            if (fillPct >= 0) {
                ctx.roundRect(barX, barY, barW * fillPct, barH, 1 * p);
            } else {
                // Animated shimmer: sliding fill block
                const phase = (this.frameTimer * 0.015) % 1;
                const blockW = barW * 0.45;
                const blockX = barX + phase * (barW + blockW) - blockW;
                const clipX = Math.max(blockX, barX);
                const clipW = Math.min(blockX + blockW, barX + barW) - clipX;
                if (clipW > 0) ctx.roundRect(clipX, barY, clipW, barH, 1 * p);
            }
            ctx.fill();
        }

        // ── Speech bubble ──
        if (this.speechTimer > 0 && this.speechBubble) {
            const alpha = this.speechTimer < 60 ? this.speechTimer / 60 : 1;
            ctx.globalAlpha = alpha;
            this._drawBubble(ctx, cx, headY - 10 * p, this.speechBubble, p);
            ctx.globalAlpha = 1;
        }

        ctx.restore();

        // Update hitbox
        this.hitbox = {
            x: cx - 10 * p, y: headY - 5 * p,
            w: 20 * p, h: (cy + 5 * p) - (headY - 5 * p)
        };
    }

    _drawBubble(ctx, x, y, text, p) {
        const maxLen = 22;
        const t = text.length > maxLen ? text.substring(0, maxLen) + '..' : text;
        ctx.font = `${5 * p}px monospace`;
        const tw = ctx.measureText(t).width;
        const pad = 4 * p;
        const bw = tw + pad * 2;
        const bh = 9 * p;
        const bx = x - bw / 2;
        const by = y - bh;

        // Shadow
        ctx.fillStyle = 'rgba(0,0,0,0.25)';
        ctx.beginPath();
        ctx.roundRect(bx + 1.5 * p, by + 1.5 * p, bw, bh, 3 * p);
        ctx.fill();

        // Bubble
        ctx.fillStyle = '#e6edf3';
        ctx.strokeStyle = '#b0b8c0';
        ctx.lineWidth = 0.8 * p;
        ctx.beginPath();
        ctx.roundRect(bx, by, bw, bh, 3 * p);
        ctx.fill();
        ctx.stroke();

        // Pointer
        ctx.fillStyle = '#e6edf3';
        ctx.beginPath();
        ctx.moveTo(x - 3 * p, by + bh);
        ctx.lineTo(x, by + bh + 3 * p);
        ctx.lineTo(x + 3 * p, by + bh);
        ctx.fill();

        ctx.fillStyle = '#1c1c1c';
        ctx.textAlign = 'center';
        ctx.fillText(t, x, by + 6.5 * p);
    }

    isHit(mx, my) {
        const h = this.hitbox;
        return mx >= h.x && mx <= h.x + h.w && my >= h.y && my <= h.y + h.h;
    }

    isRunHit(mx, my) {
        const h = this.runHitbox;
        return h && mx >= h.x && mx <= h.x + h.w && my >= h.y && my <= h.y + h.h;
    }
}

// ── Scene: Warm Wooden Floor ─────────────────────────────────────────
function drawFloor(ctx, w, h, s) {
    // Fill with base floor color
    ctx.fillStyle = C.floorA;
    ctx.fillRect(0, 0, w, h);

    const tileSize = 16 * s;
    for (let ty = 0; ty < h; ty += tileSize) {
        for (let tx = 0; tx < w; tx += tileSize) {
            const alt = ((Math.floor(tx / tileSize) + Math.floor(ty / tileSize)) % 2 === 0);
            ctx.fillStyle = alt ? C.floorA : C.floorB;
            ctx.fillRect(tx, ty, tileSize, tileSize);

            // Wood grain lines
            ctx.fillStyle = C.floorGrain;
            ctx.globalAlpha = 0.15;
            for (let g = 0; g < 3; g++) {
                const gy = ty + (g + 1) * tileSize / 4;
                ctx.fillRect(tx + 1 * s, gy, tileSize - 2 * s, 0.5 * s);
            }
            // Random knots
            if ((tx * 7 + ty * 13) % 97 < 8) {
                ctx.beginPath();
                ctx.arc(tx + tileSize / 2, ty + tileSize / 2, 1 * s, 0, Math.PI * 2);
                ctx.fill();
            }
            ctx.globalAlpha = 1;
        }
    }
}

// ── Scene: Wall ──────────────────────────────────────────────────────
function drawWall(ctx, w, wallH, s) {
    // Wall gradient
    ctx.fillStyle = C.wallDark;
    ctx.fillRect(0, 0, w, wallH);
    // Lighter middle band
    ctx.fillStyle = C.wallMid;
    ctx.fillRect(0, wallH * 0.3, w, wallH * 0.4);
    // Baseboard
    px(ctx, 0, wallH - 2 * s, w, 2 * s, C.baseboard);
    px(ctx, 0, wallH - 3 * s, w, 1 * s, '#2a3848');
}

// ── Scene: Bookshelf ─────────────────────────────────────────────────
function drawBookshelf(ctx, x, y, s) {
    const sw = 22 * s, sh = 16 * s;
    // Back
    px(ctx, x, y, sw, sh, C.shelfDark);
    // Shelves
    px(ctx, x, y, sw, 1.5 * s, C.shelfWood);
    px(ctx, x, y + sh / 2, sw, 1.5 * s, C.shelfWood);
    px(ctx, x, y + sh - 1.5 * s, sw, 1.5 * s, C.shelfWood);
    // Sides
    px(ctx, x, y, 1.5 * s, sh, C.shelfWood);
    px(ctx, x + sw - 1.5 * s, y, 1.5 * s, sh, C.shelfWood);
    // Books on top shelf
    for (let i = 0; i < 5; i++) {
        const bx = x + 2 * s + i * 3.5 * s;
        const bh = 4 * s + (i % 3) * s;
        px(ctx, bx, y + 1.5 * s + (6 * s - bh), 2.5 * s, bh, C.bookColors[i % C.bookColors.length]);
    }
    // Books on bottom shelf
    for (let i = 0; i < 4; i++) {
        const bx = x + 3 * s + i * 4 * s;
        const bh = 4 * s + (i % 2) * 1.5 * s;
        px(ctx, bx, y + sh / 2 + 1.5 * s + (6 * s - bh), 3 * s, bh, C.bookColors[(i + 3) % C.bookColors.length]);
    }
}

// ── Scene: Potted Plant ──────────────────────────────────────────────
function drawPlant(ctx, x, y, s) {
    // Pot
    px(ctx, x - 3 * s, y, 6 * s, 5 * s, C.potBrown);
    px(ctx, x - 2 * s, y - 0.5 * s, 4 * s, 1 * s, '#9a7838');
    // Rim
    px(ctx, x - 3.5 * s, y - 1 * s, 7 * s, 1.5 * s, C.potBrown);
    // Leaves (triangle clusters)
    const leafY = y - 1 * s;
    px(ctx, x - 2 * s, leafY - 5 * s, 4 * s, 5 * s, C.plantGreen);
    px(ctx, x - 4 * s, leafY - 3 * s, 3 * s, 3 * s, C.plantDark);
    px(ctx, x + 1 * s, leafY - 4 * s, 3 * s, 4 * s, C.plantDark);
    // Leaf tips
    px(ctx, x - 1 * s, leafY - 7 * s, 2 * s, 2 * s, C.plantGreen);
    px(ctx, x + 2 * s, leafY - 6 * s, 2 * s, 2 * s, C.plantGreen);
    px(ctx, x - 3 * s, leafY - 5 * s, 2 * s, 2 * s, '#50a050');
}

// ── Scene: Desk ──────────────────────────────────────────────────────
function drawDesk(ctx, x, y, s, role) {
    const dw = 32 * s, dh = 5 * s;
    const dx = x - dw / 2;

    // Desk top surface
    px(ctx, dx, y, dw, 1 * s, C.deskEdge);
    px(ctx, dx, y + 1 * s, dw, 1.5 * s, C.deskTop);

    // Desk front panel
    px(ctx, dx, y + 2.5 * s, dw, 8 * s, C.deskFront);
    // Front panel edge lines
    px(ctx, dx, y + 2.5 * s, dw, 0.5 * s, C.deskEdge);
    // Drawer lines
    px(ctx, dx + 2 * s, y + 5 * s, dw / 2 - 3 * s, 0.5 * s, C.deskSide);
    // Drawer handles
    px(ctx, dx + dw / 4, y + 6.5 * s, 2 * s, 1 * s, C.deskHandle);

    // Legs
    px(ctx, dx + 1 * s, y + 10.5 * s, 2 * s, 5 * s, C.deskSide);
    px(ctx, dx + dw - 3 * s, y + 10.5 * s, 2 * s, 5 * s, C.deskSide);

    // Keyboard on desk
    px(ctx, x - 5 * s, y - 1 * s, 10 * s, 3 * s, '#404040');
    px(ctx, x - 4.5 * s, y - 0.5 * s, 9 * s, 2 * s, '#555');
    // Key lines
    for (let i = 0; i < 4; i++) {
        px(ctx, x - 4 * s + i * 2.5 * s, y + 0.2 * s, 1.5 * s, 0.5 * s, '#666');
    }

    // Mouse
    px(ctx, x + 8 * s, y, 2 * s, 2.5 * s, '#505050');
    px(ctx, x + 8.3 * s, y + 0.3 * s, 1.4 * s, 1 * s, '#606060');

    // Coffee mug
    px(ctx, x - 12 * s, y - 2 * s, 3 * s, 3 * s, C.mugWhite);
    px(ctx, x - 9 * s, y - 1 * s, 1.5 * s, 1.5 * s, C.mugHandle);
    // Coffee inside
    px(ctx, x - 11.5 * s, y - 1.5 * s, 2 * s, 0.5 * s, '#5a3a1a');
}

// ── Scene: Monitor on Desk ───────────────────────────────────────────
function drawMonitor(ctx, x, y, s, glowColor, role, frame) {
    // Monitor frame
    const mw = 14 * s, mh = 11 * s;
    const mx = x - mw / 2, my = y - mh;

    px(ctx, mx, my, mw, mh, C.monFrame);
    // Screen
    px(ctx, mx + 1.5 * s, my + 1.5 * s, mw - 3 * s, mh - 3 * s, C.monScreen);

    // Screen content glow
    ctx.globalAlpha = 0.15;
    px(ctx, mx + 1.5 * s, my + 1.5 * s, mw - 3 * s, mh - 3 * s, glowColor);
    ctx.globalAlpha = 1;

    // Screen content based on role
    const sx = mx + 2.5 * s, sy = my + 2.5 * s;
    const sw = mw - 5 * s, sh = mh - 5 * s;

    // Slow down all screen animations — use slowFrame (updates ~every 30 frames = 0.5s)
    const slowFrame = Math.floor(frame / 30);

    if (role === 'data') {
        // Bar chart — smooth animated bars
        for (let i = 0; i < 4; i++) {
            const target = (2 + (i * 3 + slowFrame) % 5) * s;
            const barH = target;
            ctx.globalAlpha = 0.8;
            px(ctx, sx + i * 2.5 * s, sy + sh - barH, 1.8 * s, barH, glowColor);
            ctx.globalAlpha = 1;
        }
    } else if (role === 'news') {
        // Text lines — static
        for (let i = 0; i < 4; i++) {
            const lw = (5 + (i * 2) % 4) * s;
            ctx.globalAlpha = 0.6;
            px(ctx, sx, sy + i * 2 * s, lw, 0.8 * s, glowColor);
            ctx.globalAlpha = 1;
        }
    } else if (role === 'signal') {
        // Waveform — smooth slow wave
        ctx.strokeStyle = glowColor;
        ctx.lineWidth = 0.8 * s;
        ctx.globalAlpha = 0.8;
        ctx.beginPath();
        for (let i = 0; i < 8; i++) {
            const wx = sx + i * sw / 8;
            const wy = sy + sh / 2 + Math.sin((i + slowFrame) * 0.8) * 2 * s;
            if (i === 0) ctx.moveTo(wx, wy); else ctx.lineTo(wx, wy);
        }
        ctx.stroke();
        ctx.globalAlpha = 1;
    } else {
        // Global: mini world dots — steady glow
        ctx.fillStyle = glowColor;
        ctx.globalAlpha = 0.5;
        const dots = [[0.2,0.3],[0.5,0.2],[0.7,0.4],[0.3,0.6],[0.8,0.6],[0.5,0.7]];
        dots.forEach(([dx,dy]) => {
            ctx.beginPath();
            ctx.arc(sx + dx * sw, sy + dy * sh, 0.8 * s, 0, Math.PI * 2);
            ctx.fill();
        });
        ctx.globalAlpha = 1;
    }

    // Stand
    px(ctx, x - 1.5 * s, y, 3 * s, 2 * s, C.monStand);
    px(ctx, x - 3 * s, y + 1.5 * s, 6 * s, 1 * s, C.monStand);
}

// ── Scene: Chair ─────────────────────────────────────────────────────
function drawChair(ctx, x, y, s) {
    // Legs
    px(ctx, x - 4 * s, y + 5 * s, 1 * s, 3 * s, C.chairLeg);
    px(ctx, x + 3 * s, y + 5 * s, 1 * s, 3 * s, C.chairLeg);
    // Seat
    px(ctx, x - 5 * s, y + 3 * s, 10 * s, 3 * s, C.chairSeat);
    px(ctx, x - 5 * s, y + 3 * s, 10 * s, 0.8 * s, '#c8a878');
    // Back
    px(ctx, x - 4 * s, y - 6 * s, 8 * s, 9 * s, C.chairBack);
    px(ctx, x - 4 * s, y - 6 * s, 8 * s, 1 * s, '#b89060');
}

// ── Scene: Wall Clock ────────────────────────────────────────────────
function drawClock(ctx, x, y, s, frame) {
    ctx.fillStyle = '#e0d8c8';
    ctx.beginPath();
    ctx.arc(x, y, 5 * s, 0, Math.PI * 2);
    ctx.fill();
    ctx.strokeStyle = '#806040';
    ctx.lineWidth = 1 * s;
    ctx.stroke();
    // Face
    ctx.fillStyle = '#f8f0e0';
    ctx.beginPath();
    ctx.arc(x, y, 4 * s, 0, Math.PI * 2);
    ctx.fill();
    // Hands
    const angle = (frame * 0.005) % (Math.PI * 2);
    ctx.strokeStyle = '#333';
    ctx.lineWidth = 0.7 * s;
    ctx.beginPath();
    ctx.moveTo(x, y);
    ctx.lineTo(x + Math.cos(angle) * 3 * s, y + Math.sin(angle) * 3 * s);
    ctx.stroke();
    ctx.beginPath();
    ctx.moveTo(x, y);
    ctx.lineTo(x + Math.cos(angle * 0.08) * 2 * s, y + Math.sin(angle * 0.08) * 2 * s);
    ctx.stroke();
    // Center dot
    ctx.fillStyle = '#333';
    ctx.beginPath();
    ctx.arc(x, y, 0.8 * s, 0, Math.PI * 2);
    ctx.fill();
}

// ── Scene: Framed Picture ────────────────────────────────────────────
function drawPicture(ctx, x, y, s) {
    const fw = 16 * s, fh = 10 * s;
    // Frame
    px(ctx, x, y, fw, fh, '#6e4830');
    px(ctx, x + 1 * s, y + 1 * s, fw - 2 * s, fh - 2 * s, '#80c0e0');
    // Mountains
    ctx.fillStyle = '#408040';
    ctx.beginPath();
    ctx.moveTo(x + 1 * s, y + fh - 2 * s);
    ctx.lineTo(x + 5 * s, y + 3 * s);
    ctx.lineTo(x + 9 * s, y + 5 * s);
    ctx.lineTo(x + 13 * s, y + 2 * s);
    ctx.lineTo(x + fw - 1 * s, y + fh - 2 * s);
    ctx.fill();
    // Sun
    ctx.fillStyle = '#f0d040';
    ctx.beginPath();
    ctx.arc(x + 12 * s, y + 3 * s, 1.5 * s, 0, Math.PI * 2);
    ctx.fill();
}

// ── Main Trading Floor Class ─────────────────────────────────────────
class TradingFloor {
    constructor(canvas, marketCodes) {
        this.canvas = canvas;
        this.marketCodes = marketCodes;
        this.agentStatus = {};
        this.allChars = [];       // flat list of all characters
        this.marketChars = {};    // code → AgentCharacter
        this.globalChar = null;
        this.running = false;
        this.globalFrame = 0;
        this.popup = document.getElementById('agent-popup');
        window.tradingFloor = this;
        this._initCanvas();
        this._initCharacters();
        this._setupEvents();
    }

    _initCanvas() {
        this._resizeCanvas = () => {
            const r = this.canvas.parentElement.getBoundingClientRect();
            this.canvas.width = r.width * 2;
            this.canvas.height = r.height * 2;
            this.canvas.style.width = r.width + 'px';
            this.canvas.style.height = r.height + 'px';
        };
        this._resizeCanvas();
        window.addEventListener('resize', this._resizeCanvas);
        // Also observe parent size changes (e.g. chat panel open/close)
        if (typeof ResizeObserver !== 'undefined') {
            new ResizeObserver(() => this._resizeCanvas()).observe(this.canvas.parentElement);
        }
    }

    _initCharacters() {
        // 13 market agents + 1 global = 14 characters
        for (const code of this.marketCodes) {
            const c = new AgentCharacter(`${code}_market`, code, 0, 0);
            this.marketChars[code] = c;
            this.allChars.push(c);
        }
        this.globalChar = new AgentCharacter('global', 'Global', 0, 0);
        this.allChars.push(this.globalChar);
    }

    _setupEvents() {
        this.canvas.addEventListener('dblclick', (e) => {
            const r = this.canvas.getBoundingClientRect();
            const mx = (e.clientX - r.left) * 2, my = (e.clientY - r.top) * 2;
            for (const c of this.allChars) {
                if (c.isRunHit(mx, my)) {
                    if (c.state === 'running') {
                        if (window.stopSingleAgent) window.stopSingleAgent();
                    } else {
                        if (window.runSingleAgent) window.runSingleAgent(c.name);
                    }
                    return;
                }
            }
        });
        this.canvas.addEventListener('click', (e) => {
            const r = this.canvas.getBoundingClientRect();
            const mx = (e.clientX - r.left) * 2, my = (e.clientY - r.top) * 2;
            for (const c of this.allChars) {
                if (c.isRunHit(mx, my)) {
                    return; // single click on run button does nothing
                }
                if (c.isHit(mx, my)) {
                    if (window.openAgentChat) window.openAgentChat(c.name);
                    return;
                }
            }
            this._hidePopup();
        });
        this.canvas.addEventListener('mousemove', (e) => {
            const r = this.canvas.getBoundingClientRect();
            const mx = (e.clientX - r.left) * 2, my = (e.clientY - r.top) * 2;
            let hit = false;
            for (const c of this.allChars) {
                if (c.isRunHit(mx, my)) {
                    this._hidePopup();
                    this.canvas.style.cursor = 'pointer';
                    hit = true; break;
                }
                if (c.isHit(mx, my)) {
                    this._showPopup(c, e.clientX, e.clientY);
                    this.canvas.style.cursor = 'pointer';
                    hit = true; break;
                }
            }
            if (!hit) { this._hidePopup(); this.canvas.style.cursor = 'default'; }
        });
    }

    _showPopup(char, x, y) {
        const p = this.popup;
        let state, progress, duration, records, lastRun, title;

        if (char.name === 'global') {
            // Global agent has a direct status entry
            const st = this.agentStatus['global'] || {};
            state = st.state || 'idle';
            progress = st.progress || (st.last_run ? 'Done' : 'Not yet run');
            duration = st.duration_s != null ? `${st.duration_s}s` : '-';
            records = st.records != null ? st.records : '-';
            lastRun = st.last_run || '-';
            title = 'Global Strategist';
        } else {
            // Market agent: aggregate data + news + signal sub-agents
            const code = char.name.replace('_market', '');
            title = `${code} Market Agent`;
            const data = this.agentStatus[`${code}_data`] || {};
            const news = this.agentStatus[`${code}_news`] || {};
            const signal = this.agentStatus[`${code}_signal`] || {};
            const subStates = [data.state, news.state, signal.state].filter(Boolean);

            if (subStates.some(s => s === 'running')) state = 'running';
            else if (subStates.some(s => s === 'failed')) state = 'failed';
            else if (subStates.length > 0 && subStates.every(s => s === 'completed')) state = 'completed';
            else if (subStates.some(s => s === 'waiting')) state = 'waiting';
            else state = 'idle';

            const totalRecords = [data.records, news.records, signal.records]
                .filter(r => r != null).reduce((a, b) => a + b, 0);
            const totalDuration = [data.duration_s, news.duration_s, signal.duration_s]
                .filter(d => d != null).reduce((a, b) => a + b, 0);
            const lastRunArr = [data.last_run, news.last_run, signal.last_run].filter(Boolean);

            progress = state === 'completed' ? `Data ✓  News ✓  Signal ✓` :
                       state === 'idle' ? 'Not yet run' : state;
            duration = totalDuration ? `${totalDuration.toFixed(1)}s` : '-';
            records = totalRecords || '-';
            lastRun = lastRunArr.length ? lastRunArr.sort().pop().slice(0, 19) : '-';
        }

        document.getElementById('popup-title').textContent = title;
        document.getElementById('popup-state').textContent = state;
        document.getElementById('popup-progress').textContent = progress;
        document.getElementById('popup-duration').textContent = duration;
        document.getElementById('popup-records').textContent = records;
        document.getElementById('popup-lastrun').textContent = lastRun;
        p.style.left = `${x - p.parentElement.getBoundingClientRect().left + 10}px`;
        p.style.top = `${y - p.parentElement.getBoundingClientRect().top - 100}px`;
        p.classList.add('visible');
    }

    _hidePopup() { this.popup.classList.remove('visible'); }

    start() { this.running = true; this.fetchStatus(); this._loop(); }
    stop() { this.running = false; }

    async fetchStatus() {
        try {
            const r = await fetch('/api/agent-status');
            const d = await r.json();
            this.agentStatus = d.agents || {};
            this._updateFromStatus(d);
        } catch (e) {}
    }

    _updateFromStatus(data) {
        const agents = data.agents || {}, pipeline = data.pipeline || {};

        // Update global character
        if (agents['global']) {
            this.globalChar.setState(agents['global'].state, agents['global'].progress);
        }

        // Derive unified market character state from data+news+signal
        for (const code of this.marketCodes) {
            const ch = this.marketChars[code];
            if (!ch) continue;
            const keys = [`${code}_data`, `${code}_news`, `${code}_signal`];
            const present = keys.filter(k => agents[k]);
            const states = present.map(k => agents[k].state);
            let combinedState, combinedProgress;
            if (states.some(s => s === 'running')) {
                combinedState = 'running';
                const running = present.find(k => agents[k].state === 'running');
                combinedProgress = running ? agents[running].progress : 'Working...';
            } else if (states.some(s => s === 'failed')) {
                combinedState = 'failed';
                combinedProgress = 'Failed';
            } else if (present.length > 0 && states.every(s => s === 'completed')) {
                combinedState = 'completed';
                combinedProgress = 'Done';
            } else if (states.some(s => s === 'waiting')) {
                combinedState = 'waiting';
                combinedProgress = 'Waiting...';
            } else {
                combinedState = 'idle';
                combinedProgress = '';
            }
            ch.setState(combinedState, combinedProgress);
        }

        // Pipeline progress
        const comp = pipeline.completed_agents || 0, tot = pipeline.total_agents || 0;
        const state = pipeline.state || '';
        const pt = document.getElementById('pipeline-progress-text');
        const pb = document.getElementById('pipeline-progress-bar');
        if (pt) {
            if (state === 'running') pt.textContent = `Running ${comp}/${tot}`;
            else if (state === 'completed') pt.textContent = `Done ${comp}/${tot}`;
            else if (state === 'completed_with_errors') pt.textContent = `Done (errors) ${comp}/${tot}`;
            else if (state === 'failed') pt.textContent = `Failed ${comp}/${tot}`;
            else pt.textContent = 'Ready';
        }
        if (pb) {
            if (state === 'running' && tot > 0) {
                pb.style.width = `${comp / tot * 100}%`;
                pb.style.background = '#58a6ff';
            } else if (state === 'completed' || state === 'completed_with_errors') {
                pb.style.width = '100%';
                pb.style.background = state === 'completed' ? '#3fb950' : '#d29922';
            } else if (state === 'failed') {
                pb.style.width = '100%';
                pb.style.background = '#f85149';
            } else {
                pb.style.width = '0%';
                pb.style.background = '#58a6ff';
            }
        }
    }

    _loop() {
        if (!this.running) return;
        this.globalFrame++;
        this._draw();
        requestAnimationFrame(() => this._loop());
    }

    _draw() {
        const canvas = this.canvas, ctx = canvas.getContext('2d');
        const w = canvas.width, h = canvas.height;
        const n = this.marketCodes.length; // 13
        // Layout: 7 columns x 2 rows for markets, global agent in center-bottom
        const cols = 7;
        const rows = Math.ceil(n / cols); // 2

        // Scale to fit all desks
        const s = Math.min(w / (cols * 32 + 20), h / ((rows + 1) * 55 + 40));
        const wallH = 25 * s;
        const deskSpacingX = (w - 20 * s) / cols;
        const deskSpacingY = 52 * s;
        const startY = wallH + 18 * s;

        ctx.clearRect(0, 0, w, h);
        drawFloor(ctx, w, h, s);
        drawWall(ctx, w, wallH, s);

        // Wall decorations — spread across width
        const wallSlots = Math.floor(w / (30 * s));
        for (let i = 0; i < wallSlots; i++) {
            const wx = (i + 0.5) * (w / wallSlots);
            if (i % 3 === 0) drawBookshelf(ctx, wx - 11 * s, wallH - 18 * s, s);
            else if (i % 3 === 1) drawPicture(ctx, wx - 8 * s, wallH - 14 * s, s);
            else drawClock(ctx, wx, wallH - 9 * s, s, this.globalFrame);
        }

        // Title
        ctx.fillStyle = '#a0b0c0';
        ctx.font = `bold ${6 * s}px monospace`;
        ctx.textAlign = 'center';
        ctx.fillText('Global Tech Market Trading Floor', w / 2, wallH - 20 * s);

        // Draw market agents in grid
        for (let i = 0; i < n; i++) {
            const col = i % cols;
            const row = Math.floor(i / cols);
            const cx = 10 * s + deskSpacingX * (col + 0.5);
            const cy = startY + row * deskSpacingY;
            const code = this.marketCodes[i];
            const ch = this.marketChars[code];

            drawChair(ctx, cx, cy + 2 * s, s);
            drawDesk(ctx, cx, cy + 6 * s, s, 'market');
            drawMonitor(ctx, cx, cy + 4 * s, s, AGENT_STYLES.market.monGlow, 'data', this.globalFrame);

            ch.x = cx / s;
            ch.y = cy / s;
            ch.update();
            ch.draw(ctx, s);
        }

        // Global agent — center of bottom area
        const globalY = startY + rows * deskSpacingY;
        const globalX = w / 2;
        drawChair(ctx, globalX, globalY + 2 * s, s);
        drawDesk(ctx, globalX, globalY + 6 * s, s, 'global');
        drawMonitor(ctx, globalX, globalY + 4 * s, s, AGENT_STYLES.global.monGlow, 'global', this.globalFrame);
        drawMonitor(ctx, globalX + 9 * s, globalY + 4 * s, s, C.blue, 'data', this.globalFrame);

        this.globalChar.x = globalX / s;
        this.globalChar.y = globalY / s;
        this.globalChar.update();
        this.globalChar.draw(ctx, s);

        // Plants between rows
        drawPlant(ctx, 6 * s, startY + deskSpacingY - 5 * s, s);
        drawPlant(ctx, w - 6 * s, startY + deskSpacingY - 5 * s, s);
    }
}
