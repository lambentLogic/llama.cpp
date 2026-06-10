// cartridge-cli: minimal demonstrator for KV "cartridges" (HazyResearch Cartridges paper).
//
// A cartridge is a pre-trained KV cache prefix stored as a llama.cpp state-seq file.
// This tool has two modes:
//
//   --save-cartridge OUT   tokenize the prompt, decode it into seq 0, then
//                          llama_state_seq_save_file(OUT). Produces a ground-truth
//                          state-seq file (used by the format / equivalence tests and as a
//                          quick way to make a "cartridge" from real text).
//
//   --cartridge IN         llama_state_seq_load_file(IN) into seq 0 (cells 0..N-1), then
//                          decode the prompt at positions starting N and generate.
//
// The cartridge path REQUIRES the runtime layout the converter targets:
//   flash attention enabled (V untransposed, v_trans=0) and a unified KV cache
//   (n_stream=1). Both are forced on here.
//
// Intentionally dependency-light: uses only the public C API (no common/ framework).

#include "llama.h"

#include <cstdio>
#include <cstring>
#include <string>
#include <vector>

static void print_usage(const char * argv0) {
    printf("\nusage: %s -m model.gguf [options] -p \"prompt\"\n", argv0);
    printf("\noptions:\n");
    printf("  -m PATH               model gguf (required)\n");
    printf("  -p TEXT               prompt text\n");
    printf("  -n N                  number of tokens to generate (default 64)\n");
    printf("  -c N                  context size (default 4096)\n");
    printf("  -ngl N                gpu layers to offload (default 99)\n");
    printf("  --cache-type f16|f32  KV cache dtype (default f16)\n");
    printf("  --cartridge PATH      load a cartridge prefix, then generate from the prompt\n");
    printf("  --save-cartridge PATH decode the prompt and save it as a cartridge file\n");
    printf("  --no-bos              do not prepend BOS when tokenizing the prompt\n");
    printf("\n");
}

static enum ggml_type parse_cache_type(const std::string & s) {
    if (s == "f16") return GGML_TYPE_F16;
    if (s == "f32") return GGML_TYPE_F32;
    fprintf(stderr, "error: unsupported --cache-type '%s' (use f16 or f32)\n", s.c_str());
    return GGML_TYPE_F16;
}

int main(int argc, char ** argv) {
    std::string model_path;
    std::string prompt = "Hello my name is";
    std::string cartridge_in;
    std::string cartridge_out;
    int n_predict = 64;
    int n_ctx = 4096;
    int ngl = 99;
    bool add_bos = true;
    enum ggml_type cache_type = GGML_TYPE_F16;

    for (int i = 1; i < argc; i++) {
        std::string a = argv[i];
        auto next = [&](const char * name) -> const char * {
            if (i + 1 >= argc) { fprintf(stderr, "error: %s requires an argument\n", name); exit(1); }
            return argv[++i];
        };
        if      (a == "-m")               model_path    = next("-m");
        else if (a == "-p")               prompt        = next("-p");
        else if (a == "-n")               n_predict     = std::stoi(next("-n"));
        else if (a == "-c")               n_ctx         = std::stoi(next("-c"));
        else if (a == "-ngl")             ngl           = std::stoi(next("-ngl"));
        else if (a == "--cache-type")     cache_type    = parse_cache_type(next("--cache-type"));
        else if (a == "--cartridge")      cartridge_in  = next("--cartridge");
        else if (a == "--save-cartridge") cartridge_out = next("--save-cartridge");
        else if (a == "--no-bos")         add_bos       = false;
        else if (a == "-h" || a == "--help") { print_usage(argv[0]); return 0; }
        else { fprintf(stderr, "error: unknown argument '%s'\n", a.c_str()); print_usage(argv[0]); return 1; }
    }

    if (model_path.empty()) { print_usage(argv[0]); return 1; }

    ggml_backend_load_all();

    llama_model_params mparams = llama_model_default_params();
    mparams.n_gpu_layers = ngl;
    llama_model * model = llama_model_load_from_file(model_path.c_str(), mparams);
    if (!model) { fprintf(stderr, "error: failed to load model\n"); return 1; }

    const llama_vocab * vocab = llama_model_get_vocab(model);

    llama_context_params cparams = llama_context_default_params();
    cparams.n_ctx           = n_ctx;
    cparams.n_batch         = n_ctx;
    cparams.flash_attn_type = LLAMA_FLASH_ATTN_TYPE_ENABLED;  // -> v_trans = 0
    cparams.kv_unified      = true;                            // -> n_stream = 1
    cparams.type_k          = cache_type;
    cparams.type_v          = cache_type;
    cparams.no_perf         = false;

    llama_context * ctx = llama_init_from_model(model, cparams);
    if (!ctx) { fprintf(stderr, "error: failed to create context\n"); return 1; }

    llama_memory_t mem = llama_get_memory(ctx);

    // tokenize prompt
    int n_prompt = -llama_tokenize(vocab, prompt.c_str(), prompt.size(), nullptr, 0, add_bos, true);
    std::vector<llama_token> prompt_tokens(n_prompt);
    if (llama_tokenize(vocab, prompt.c_str(), prompt.size(), prompt_tokens.data(), prompt_tokens.size(), add_bos, true) < 0) {
        fprintf(stderr, "error: tokenize failed\n"); return 1;
    }

    // ----- helper: decode a vector of tokens for seq 0 starting at pos p0 -----
    auto decode_tokens = [&](const std::vector<llama_token> & toks, int p0, bool want_last_logits) -> bool {
        llama_batch batch = llama_batch_init((int32_t) toks.size(), 0, 1);
        for (size_t i = 0; i < toks.size(); ++i) {
            batch.token[i]      = toks[i];
            batch.pos[i]        = p0 + (int) i;
            batch.n_seq_id[i]   = 1;
            batch.seq_id[i][0]  = 0;
            batch.logits[i]     = (want_last_logits && i + 1 == toks.size()) ? 1 : 0;
        }
        batch.n_tokens = (int32_t) toks.size();
        bool ok = llama_decode(ctx, batch) == 0;
        llama_batch_free(batch);
        return ok;
    };

    // ===== save mode =====
    if (!cartridge_out.empty()) {
        if (!decode_tokens(prompt_tokens, 0, false)) { fprintf(stderr, "error: decode failed\n"); return 1; }
        size_t written = llama_state_seq_save_file(ctx, cartridge_out.c_str(), 0,
                                                   prompt_tokens.data(), prompt_tokens.size());
        if (written == 0) { fprintf(stderr, "error: failed to save cartridge\n"); return 1; }
        fprintf(stderr, "saved cartridge '%s' (%d tokens, %zu bytes)\n",
                cartridge_out.c_str(), (int) prompt_tokens.size(), written);
        llama_free(ctx); llama_model_free(model);
        return 0;
    }

    // ===== generate mode =====
    int n_past = 0;
    if (!cartridge_in.empty()) {
        std::vector<llama_token> dummy(n_ctx);
        size_t n_tok_out = 0;
        size_t read = llama_state_seq_load_file(ctx, cartridge_in.c_str(), 0,
                                                dummy.data(), dummy.size(), &n_tok_out);
        if (read == 0) { fprintf(stderr, "error: failed to load cartridge\n"); return 1; }
        n_past = llama_memory_seq_pos_max(mem, 0) + 1;  // cells occupy 0..n_past-1
        fprintf(stderr, "loaded cartridge '%s': %d KV cells (prefix tokens recorded: %zu)\n",
                cartridge_in.c_str(), n_past, n_tok_out);
    }

    // decode the prompt after the cartridge prefix
    if (!decode_tokens(prompt_tokens, n_past, true)) { fprintf(stderr, "error: decode failed\n"); return 1; }
    int n_cur = n_past + (int) prompt_tokens.size();

    // greedy sampler
    llama_sampler * smpl = llama_sampler_chain_init(llama_sampler_chain_default_params());
    llama_sampler_chain_add(smpl, llama_sampler_init_greedy());

    // echo prompt
    for (auto id : prompt_tokens) {
        char buf[256];
        int n = llama_token_to_piece(vocab, id, buf, sizeof(buf), 0, true);
        if (n > 0) printf("%.*s", n, buf);
    }
    fflush(stdout);

    for (int i = 0; i < n_predict; ++i) {
        llama_token id = llama_sampler_sample(smpl, ctx, -1);
        if (llama_vocab_is_eog(vocab, id)) break;

        char buf[256];
        int n = llama_token_to_piece(vocab, id, buf, sizeof(buf), 0, true);
        if (n > 0) printf("%.*s", n, buf);
        fflush(stdout);

        std::vector<llama_token> one{ id };
        if (!decode_tokens(one, n_cur, true)) { fprintf(stderr, "\nerror: decode failed\n"); break; }
        n_cur++;
    }
    printf("\n");

    llama_sampler_free(smpl);
    llama_free(ctx);
    llama_model_free(model);
    return 0;
}
