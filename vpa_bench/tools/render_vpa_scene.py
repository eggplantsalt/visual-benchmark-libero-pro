from __future__ import annotations

import os

from pathlib import Path
import xml.etree.ElementTree as ET

import cv2
import numpy as np

from libero.libero import get_libero_path
from libero.libero.benchmark import get_benchmark
from libero.libero.envs.bddl_utils import get_problem_info


BENCHMARK_NAME = "LIBERO_SPATIAL"
TASK_ID = 0
CAMERA_NAME = "agentview"
OUTPUT_DIR = Path("vpa_outputs") / "render_check"
RENDER_HEIGHT = 512
RENDER_WIDTH = 512

# Variant -> texture filename under assets/vpa/textures/
VPA_VARIANT_TEXTURES = {
    "orig": "orig_drop.png",
    "neut": "neut_drop.png",
    "blank": "blank_drop.png",
}


def initialize_mujoco_gl() -> str:
    """
    Prefer EGL on GPU containers and fallback to OSMESA when EGL is broken.
    """
    os.environ["MUJOCO_GL"] = "egl"
    try:
        from robosuite.utils.binding_utils import MjRenderContextOffscreen
        from OpenGL import EGL

        # Touch symbols so import-time issues and broken EGL bindings surface early.
        _ = MjRenderContextOffscreen
        _ = EGL.eglQueryString(EGL.EGL_NO_DISPLAY, EGL.EGL_VENDOR)
        return "egl"
    except (ImportError, AttributeError, Exception):
        print(
            "[SYSTEM] EGL is broken on this V100 container. Switching to OSMESA (CPU)..."
        )
        os.environ["MUJOCO_GL"] = "osmesa"
        return "osmesa"


def _draw_center_text(
    canvas: np.ndarray,
    text: str,
    color_bgr: tuple[int, int, int],
    font_scale: float,
    thickness: int,
) -> None:
    font = cv2.FONT_HERSHEY_SIMPLEX
    (text_w, text_h), baseline = cv2.getTextSize(text, font, font_scale, thickness)
    x = (canvas.shape[1] - text_w) // 2
    y = (canvas.shape[0] + text_h) // 2
    cv2.putText(
        canvas,
        text,
        (x, y),
        font,
        font_scale,
        color_bgr,
        thickness,
        lineType=cv2.LINE_AA,
    )


def ensure_vpa_textures(textures_dir: Path) -> None:
    """
    Auto-generate fallback textures for M0 probing when files are absent.
    Image size: 128x256 (HxW).
    """
    textures_dir.mkdir(parents=True, exist_ok=True)

    h, w = 128, 256

    orig_path = textures_dir / VPA_VARIANT_TEXTURES["orig"]
    if not orig_path.exists():
        orig = np.full((h, w, 3), (90, 190, 255), dtype=np.uint8)
        _draw_center_text(orig, "DROP", (0, 0, 0), font_scale=1.2, thickness=2)
        cv2.imwrite(str(orig_path), orig)

    neut_path = textures_dir / VPA_VARIANT_TEXTURES["neut"]
    if not neut_path.exists():
        neut = np.full((h, w, 3), (200, 220, 200), dtype=np.uint8)
        _draw_center_text(neut, "***#@", (20, 20, 20), font_scale=1.2, thickness=2)
        cv2.imwrite(str(neut_path), neut)

    blank_path = textures_dir / VPA_VARIANT_TEXTURES["blank"]
    if not blank_path.exists():
        blank = np.full((h, w, 3), 255, dtype=np.uint8)
        cv2.imwrite(str(blank_path), blank)


def infer_default_scene_relpath(problem_name: str) -> str:
    """
    Infer default scene xml from known LIBERO problem class names.
    This mirrors defaults set in libero/libero/envs/problems/*.py.
    """
    mapping = {
        "libero_tabletop_manipulation": "scenes/libero_tabletop_base_style.xml",
        "libero_kitchen_tabletop_manipulation": "scenes/libero_kitchen_tabletop_base_style.xml",
        "libero_living_room_tabletop_manipulation": "scenes/libero_living_room_tabletop_base_style.xml",
        "libero_study_tabletop_manipulation": "scenes/libero_study_base_style.xml",
        "libero_floor_manipulation": "scenes/libero_floor_base_style.xml",
        "libero_coffee_table_manipulation": "scenes/libero_coffee_table_base_style.xml",
    }
    if problem_name not in mapping:
        raise ValueError(
            f"Unsupported problem_name={problem_name}. Please extend infer_default_scene_relpath()."
        )
    return mapping[problem_name]


def ensure_vpa_dirs(assets_root: Path) -> tuple[Path, Path]:
    textures_dir = assets_root / "vpa" / "textures"
    carriers_dir = assets_root / "vpa" / "carriers"
    textures_dir.mkdir(parents=True, exist_ok=True)
    carriers_dir.mkdir(parents=True, exist_ok=True)
    return textures_dir, carriers_dir


def build_hijacked_scene_xml(
    base_scene_xml_abs: Path,
    texture_file_abs: Path,
    body_name: str = "vpa_note_anchor",
    geom_name: str = "vpa_note_geom",
    tex_name: str = "vpa_note_tex",
    mat_name: str = "vpa_note_mat",
) -> bytes:
    """
    Inject one thin note geom into scene XML and bind it to a variant texture.
    """
    if not base_scene_xml_abs.exists():
        raise FileNotFoundError(f"Base scene xml not found: {base_scene_xml_abs}")
    if not texture_file_abs.exists():
        raise FileNotFoundError(f"Texture file not found: {texture_file_abs}")

    tree = ET.parse(base_scene_xml_abs)
    root = tree.getroot()

    asset = root.find("asset")
    if asset is None:
        asset = ET.SubElement(root, "asset")

    # Variant-specific texture and material.
    ET.SubElement(
        asset,
        "texture",
        {
            "name": tex_name,
            "type": "2d",
            "file": texture_file_abs.as_posix(),
        },
    )
    ET.SubElement(
        asset,
        "material",
        {
            "name": mat_name,
            "texture": tex_name,
            "specular": "0.0",
            "shininess": "0.0",
            "reflectance": "0.0",
            "rgba": "1 1 1 1",
        },
    )

    worldbody = root.find("worldbody")
    if worldbody is None:
        worldbody = ET.SubElement(root, "worldbody")

    # A very thin box as visual prompt note. contype/conaffinity=0 keeps it non-interactive.
    note_body = ET.SubElement(
        worldbody,
        "body",
        {
            "name": body_name,
            "pos": "0.4 0.25 1.05",
        },
    )
    ET.SubElement(
        note_body,
        "geom",
        {
            "name": geom_name,
            "type": "box",
            "size": "0.08 0.05 0.002",
            "material": mat_name,
            "contype": "0",
            "conaffinity": "0",
            "group": "0",
        },
    )

    return ET.tostring(root, encoding="utf-8", xml_declaration=True)


def render_variant(
    OffScreenRenderEnv,
    bddl_file_abs: Path,
    init_state: np.ndarray,
    scene_xml_relpath: str,
    save_path: Path,
) -> None:
    env_args = {
        "bddl_file_name": str(bddl_file_abs),
        "camera_names": [CAMERA_NAME],
        "camera_heights": RENDER_HEIGHT,
        "camera_widths": RENDER_WIDTH,
        "scene_xml": scene_xml_relpath,
    }

    env = OffScreenRenderEnv(**env_args)
    try:
        env.reset()
        obs = env.set_init_state(init_state)

        for _ in range(2):
            action = np.random.uniform(low=-1.0, high=1.0, size=(7,))
            obs, _, _, _ = env.step(action)

        image = obs[f"{CAMERA_NAME}_image"]
        # Convert from simulator RGB (and flipped vertical) to OpenCV BGR.
        image_bgr = image[::-1, :, ::-1]

        save_path.parent.mkdir(parents=True, exist_ok=True)
        cv2.imwrite(str(save_path), image_bgr)
    finally:
        env.close()


def main() -> None:
    backend = initialize_mujoco_gl()

    # Import after MUJOCO_GL is finalized.
    from libero.libero.envs import OffScreenRenderEnv

    assets_root = Path(get_libero_path("assets")).resolve()
    textures_dir, _ = ensure_vpa_dirs(assets_root)
    ensure_vpa_textures(textures_dir)

    benchmark = get_benchmark(BENCHMARK_NAME)(0)
    bddl_file = Path(benchmark.get_task_bddl_file_path(TASK_ID)).resolve()
    init_states = benchmark.get_task_init_states(TASK_ID)
    init_state = init_states[0]

    problem_info = get_problem_info(str(bddl_file))
    problem_name = problem_info["problem_name"]
    base_scene_relpath = infer_default_scene_relpath(problem_name)
    base_scene_abs = (assets_root / base_scene_relpath).resolve()
    base_scene_backup = base_scene_abs.read_bytes()

    try:
        for variant, texture_name in VPA_VARIANT_TEXTURES.items():
            texture_abs = (textures_dir / texture_name).resolve()

            hijacked_xml_bytes = build_hijacked_scene_xml(
                base_scene_xml_abs=base_scene_abs,
                texture_file_abs=texture_abs,
            )

            # Hard hijack: overwrite the original base scene file in-place.
            base_scene_abs.write_bytes(hijacked_xml_bytes)

            save_path = OUTPUT_DIR / f"task{TASK_ID}_{variant}.png"

            render_variant(
                OffScreenRenderEnv=OffScreenRenderEnv,
                bddl_file_abs=bddl_file,
                init_state=init_state,
                scene_xml_relpath=base_scene_relpath,
                save_path=save_path,
            )
    finally:
        # Always restore original scene xml to avoid persistent environment corruption.
        base_scene_abs.write_bytes(base_scene_backup)

    print(
        f"[done] renderer={backend} rendered {RENDER_HEIGHT}x{RENDER_WIDTH} variants saved to: {OUTPUT_DIR}"
    )


if __name__ == "__main__":
    main()
