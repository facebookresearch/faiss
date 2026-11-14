/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include <faiss/impl/SearchParameters.h>

#include <typeinfo>

#include <faiss/IndexIDMap.h>
#include <faiss/IndexIVF.h>
#include <faiss/IndexPreTransform.h>
#include <faiss/IndexRefine.h>
#include <faiss/impl/FaissAssert.h>

namespace faiss {

SearchParameters* SearchParametersStorage::copy(
        const SearchParameters* params) {
    std::unique_ptr<SearchParameters> copied(params->clone());
    FAISS_THROW_IF_NOT_MSG(
            copied && typeid(*copied) == typeid(*params),
            "SearchParameters subclass must override clone() to preserve its "
            "concrete type when forwarding selectors");
    auto* result = copied.get();
    parameters.push_back(std::move(copied));
    return result;
}

const SearchParameters* SearchParametersStorage::with_selector(
        const Index* index,
        const SearchParameters* params,
        IDSelector* sel) {
    if (!sel || (params && params->sel == sel)) {
        return params;
    }
    SearchParameters* result;
    if (params) {
        result = copy(params);
    } else {
        // These indexes require their own parameter types even when only a
        // selector is supplied. Other indexes accept SearchParameters or
        // explicitly reject unsupported selectors/parameters.
        while (auto* idmap = dynamic_cast<const IndexIDMap*>(index)) {
            index = idmap->index;
        }
        std::unique_ptr<SearchParameters> defaults;
        if (auto* refine = dynamic_cast<const IndexRefine*>(index)) {
            auto p = std::make_unique<IndexRefineSearchParameters>();
            p->k_factor = refine->k_factor;
            defaults = std::move(p);
        } else if (dynamic_cast<const IndexPreTransform*>(index)) {
            defaults = std::make_unique<SearchParametersPreTransform>();
        } else if (auto* ivf = dynamic_cast<const IndexIVF*>(index)) {
            auto p = std::make_unique<SearchParametersIVF>();
            p->nprobe = ivf->nprobe;
            p->max_codes = ivf->max_codes;
            defaults = std::move(p);
        } else {
            defaults = std::make_unique<SearchParameters>();
        }
        result = defaults.get();
        parameters.push_back(std::move(defaults));
    }
    if (result->sel) {
        auto intersection = std::make_unique<IDSelectorAnd>(sel, result->sel);
        result->sel = intersection.get();
        selectors.push_back(std::move(intersection));
    } else {
        result->sel = sel;
    }
    return result;
}

const SearchParameters* SearchParametersStorage::translate_selectors(
        const SearchParameters* params,
        const std::vector<idx_t>& id_map) {
    if (!params) {
        return nullptr;
    }
    const SearchParameters* child = nullptr;
    auto* refine = dynamic_cast<const IndexRefineSearchParameters*>(params);
    auto* transform = dynamic_cast<const SearchParametersPreTransform*>(params);
    if (refine) {
        child = refine->base_index_params;
    } else if (transform) {
        child = transform->index_params;
    }
    const auto* translated_child = translate_selectors(child, id_map);
    // Preserve the existing explicit-IDSelectorTranslated convention. Only
    // skip translation for this map, so nested IDMaps can compose correctly.
    auto* translated = dynamic_cast<const IDSelectorTranslated*>(params->sel);
    bool translate =
            params->sel && !(translated && &translated->id_map == &id_map);
    if (!translate && child == translated_child) {
        return params;
    }
    auto* result = copy(params);
    if (translate) {
        auto selector =
                std::make_unique<IDSelectorTranslated>(id_map, params->sel);
        result->sel = selector.get();
        selectors.push_back(std::move(selector));
    }
    if (refine) {
        static_cast<IndexRefineSearchParameters*>(result)->base_index_params =
                const_cast<SearchParameters*>(translated_child);
    } else if (transform) {
        static_cast<SearchParametersPreTransform*>(result)->index_params =
                const_cast<SearchParameters*>(translated_child);
    }
    return result;
}

} // namespace faiss
