// pi extension: drive the clinical OCR bake-off harness on bigbox.
// Tools: ocr_status, ocr_leaderboard, ocr_report, ocr_run, ocr_publish.
// Location-aware: uses ~/ocr-bench directly when running on bigbox,
// otherwise shells out over `ssh bigbox` (so the same file works on the Mac).
import type { ExtensionAPI } from "@earendil-works/pi-coding-agent";
import { Type } from "typebox";
import { execFile } from "node:child_process";
import { existsSync } from "node:fs";
import { promisify } from "node:util";

const ROOT = "/home/nick/ocr-bench";
const LOCAL = existsSync(ROOT + "/overnight.sh");
const PY = `${ROOT}/.venv/bin/python3 ${ROOT}/pi_tools.py`;
const exec = promisify(execFile) as (
  cmd: string, args: string[], opts?: { timeout?: number; maxBuffer?: number }
) => Promise<{ stdout: string; stderr: string }>;

async function harness(subcmd: string, timeoutMs = 60000): Promise<string> {
  const opts = { timeout: timeoutMs, maxBuffer: 16 * 1024 * 1024 };
  try {
    if (LOCAL) {
      const r = await exec("/bin/bash", ["-lc", `${PY} ${subcmd}`], opts);
      return r.stdout + (r.stderr ? `\n[stderr]\n${r.stderr}` : "");
    }
    const r = await exec(
      "ssh", ["-o", "ConnectTimeout=10", "bigbox", `${PY} ${subcmd}`], opts);
    return r.stdout + (r.stderr ? `\n[stderr]\n${r.stderr}` : "");
  } catch (e: unknown) {
    const err = e as { stdout?: string; stderr?: string; message?: string };
    return `ERROR: ${err.message ?? String(e)}\n${err.stdout ?? ""}\n${err.stderr ?? ""}`;
  }
}

export default function (pi: ExtensionAPI) {
  pi.registerTool({
    name: "ocr_status",
    label: "OCR harness status",
    description:
      "Status of the clinical OCR bake-off harness on bigbox (~/ocr-bench): " +
      "overnight queue state, recent experiment results with exit codes, GPU " +
      "utilisation, docker containers, last git commits. Cheap; call this first.",
    parameters: Type.Object({}),
    async execute() {
      const text = await harness("status", 45000);
      return { content: [{ type: "text", text }], details: {} };
    },
  });

  pi.registerTool({
    name: "ocr_leaderboard",
    label: "OCR leaderboard",
    description:
      "Scored-model leaderboard for the OCR harness: CER (mean/median), " +
      "pages/sec, and handwriting/rotated/tables subset scores per model. " +
      "Use --query to filter by substring (e.g. 'medreal', 'lora', 'dots').",
    parameters: Type.Object({
      query: Type.String({ description: "substring filter on model name", default: "" }),
      limit: Type.Number({ description: "max rows", default: 25 }),
    }),
    async execute(toolCallId, params) {
      const q = params.query ? ` --query ${JSON.stringify(params.query)}` : "";
      const n = params.limit ? ` -n ${Math.floor(params.limit)}` : "";
      const text = await harness(`leaderboard${q}${n}`, 45000);
      return { content: [{ type: "text", text }], details: {} };
    },
  });

  pi.registerTool({
    name: "ocr_report",
    label: "OCR report",
    description:
      "Return the current write-up (reports/REPORT.md) from the OCR harness: " +
      "leaderboard, failure analysis, router study, distillation results, cost " +
      "math, run log. Truncated to `chars` (0 = full, can be long).",
    parameters: Type.Object({
      chars: Type.Number({ description: "max characters to return (0 = full)", default: 12000 }),
    }),
    async execute(toolCallId, params) {
      const text = await harness(`report --chars ${Math.floor(params.chars || 0)}`, 45000);
      return { content: [{ type: "text", text }], details: {} };
    },
  });

  pi.registerTool({
    name: "ocr_run",
    label: "Run OCR eval",
    description:
      "Launch a model evaluation on the OCR harness (vLLM + ClinOCR-Bench, 328 " +
      "clinical scans) in the background on bigbox. Returns immediately; poll " +
      "ocr_status or the log path. Results land in results/<name>/ and enter the " +
      "leaderboard after ocr_publish. Requires a free GPU (check ocr_status).",
    parameters: Type.Object({
      name: Type.String({ description: "short unique run name (used as results dir)" }),
      hf_id: Type.String({ description: "HuggingFace model id, e.g. rednote-hilab/dots.mocr" }),
      prompt: Type.String({ description: "OCR instruction prompt for the model" }),
      gpu: Type.Number({ description: "GPU index 0 or 1", default: 0 }),
      concurrency: Type.Number({ description: "parallel requests", default: 8 }),
    }),
    async execute(toolCallId, params) {
      const esc = (s: string) => `'${s.replace(/'/g, `'\\''`)}'`;
      const sub =
        `run --name ${esc(params.name)} --hf-id ${esc(params.hf_id)} ` +
        `--prompt ${esc(params.prompt)} --gpu ${Math.floor(params.gpu)} ` +
        `--concurrency ${Math.floor(params.concurrency)}`;
      const text = await harness(sub, 30000);
      return { content: [{ type: "text", text }], details: {} };
    },
  });

  pi.registerTool({
    name: "ocr_publish",
    label: "Publish OCR results",
    description:
      "Regenerate reports/REPORT.md from all results in the harness and push to " +
      "GitHub (MadCodeTX/clinical-ocr-model-bakeoff). Run after new results land. " +
      "May take ~1 minute.",
    parameters: Type.Object({}),
    async execute() {
      const text = await harness("publish", 300000);
      return { content: [{ type: "text", text }], details: {} };
    },
  });
}
