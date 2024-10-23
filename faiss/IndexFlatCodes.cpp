/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include <faiss/IndexFlatCodes.h>

#include <faiss/impl/AuxIndexStructures.h>
#include <faiss/impl/CodePacker.h>
#include <faiss/impl/DistanceComputer.h>
#include <faiss/impl/FaissAssert.h>
#include <faiss/impl/IDSelector.h>
#include <faiss/impl/ResultHandler.h>
#include <faiss/utils/extra_distances.h>

namespace faiss {

IndexFlatCodes::IndexFlatCodes(size_t code_size, idx_t d, MetricType metric)
        : Index(d, metric), code_size(code_size) {}

IndexFlatCodes::IndexFlatCodes(size_t code_size, idx_t d, bool is_include_one_attribute, MetricType metric)
        : Index(d, metric), code_size(code_size), is_include_one_attribute(is_include_one_attribute) {}

IndexFlatCodes::IndexFlatCodes(size_t code_size, idx_t d, bool is_include_two_attribute, bool mode_two, MetricType metric)
        : Index(d, metric), code_size(code_size), is_include_two_attribute(is_include_two_attribute), mode_two(mode_two) {}

IndexFlatCodes::IndexFlatCodes() : code_size(0) {}

void IndexFlatCodes::add(idx_t n, const float* x) {
    FAISS_THROW_IF_NOT(is_trained);
    FAISS_THROW_IF_NOT_MSG(is_include_one_attribute==false, "Index now has one attribute, add vector without one attribute cause conflict");
    FAISS_THROW_IF_NOT_MSG(is_include_two_attribute==false, "Index now has two attribute, add vector without two attribute cause conflict");
    if (n == 0) {
        return;
    }
    codes.resize((ntotal + n) * code_size);
    sa_encode(n, x, codes.data() + (ntotal * code_size));
    ntotal += n;
}

void IndexFlatCodes::add_with_one_attribute(idx_t n, const float* x, const float* attr) {
    FAISS_THROW_IF_NOT(is_trained);
    FAISS_THROW_IF_NOT_MSG(is_include_one_attribute, "Current Index dont't include one attribute yet !");
    if (n == 0) {
        return;
    }
    codes.resize((ntotal + n) * code_size);
    attributes.resize((ntotal + n) * attr_size);
    sa_encode(n, x, codes.data() + (ntotal * code_size));
    sa_one_attribute_encode(n, attr, attributes.data() + (ntotal * attr_size));
    ntotal += n;
}

void IndexFlatCodes::add_with_two_attribute(idx_t n, const float* x, const float* attr_first, const float* attr_second) {
    FAISS_THROW_IF_NOT(is_trained);
    FAISS_THROW_IF_NOT_MSG(is_include_two_attribute, "Current Index dont't include two attribute yet !");
    if (n == 0) {
        return;
    }
    codes.resize((ntotal + n) * code_size);
    attributes_first.resize((ntotal + n) * attr_size);
    attributes_second.resize((ntotal + n) * attr_size);
    sa_encode(n, x, codes.data() + (ntotal * code_size));
    sa_two_attribute_encode(n, attr_first, attr_second, 
                            attributes_first.data() + (ntotal * attr_size), attributes_second.data() + (ntotal * attr_size));
    ntotal += n;
}

bool IndexFlatCodes::get_is_include_one_attribute() {
    return is_include_one_attribute;
}

void IndexFlatCodes::set_is_include_one_attribute() {
    this->is_include_one_attribute = true;
}

bool IndexFlatCodes::get_is_include_two_attribute() {
    return is_include_two_attribute;
}

void IndexFlatCodes::set_is_include_two_attribute() {
    this->is_include_two_attribute = true;
}

void IndexFlatCodes::reset() {
    codes.clear();
    if (get_is_include_one_attribute()) {
        attributes.clear();
    }
    if (get_is_include_two_attribute()) {
        attributes_first.clear();
        attributes_second.clear();
    }
    ntotal = 0;
}

size_t IndexFlatCodes::sa_code_size() const {
    return code_size;
}

size_t IndexFlatCodes::sa_one_attribute_code_size() const {
    return attr_size;
}

size_t IndexFlatCodes::sa_two_attribute_code_size() const {
    return attr_size;
}

size_t IndexFlatCodes::remove_ids(const IDSelector& sel) {
    idx_t j = 0;
    for (idx_t i = 0; i < ntotal; i++) {
        if (sel.is_member(i)) {
            // should be removed
        } else {
            if (i > j) {
                memmove(&codes[code_size * j],
                        &codes[code_size * i],
                        code_size);

                if (get_is_include_one_attribute()) {
                    memmove(&attributes[attr_size * j],
                            &attributes[attr_size * i],
                            attr_size);
                }

                if (get_is_include_two_attribute()) {
                    memmove(&attributes_first[attr_size * j],
                            &attributes_first[attr_size * i],
                            attr_size);
                    memmove(&attributes_second[attr_size * j],
                            &attributes_second[attr_size * i],
                            attr_size);
                }
            }
            j++;
        }
    }
    size_t nremove = ntotal - j;
    if (nremove > 0) {
        ntotal = j;
        codes.resize(ntotal * code_size);

        if (get_is_include_one_attribute()) {
            attributes.resize(ntotal * attr_size);
        }

        if (get_is_include_two_attribute()) {
            attributes_first.resize(ntotal * attr_size);
            attributes_second.resize(ntotal * attr_size);
        }
    }
    return nremove;
}

void IndexFlatCodes::reconstruct_n(idx_t i0, idx_t ni, float* recons) const {
    FAISS_THROW_IF_NOT(ni == 0 || (i0 >= 0 && i0 + ni <= ntotal));
    sa_decode(ni, codes.data() + i0 * code_size, recons);
}

void IndexFlatCodes::reconstruct_n_one_attribute(idx_t i0, idx_t ni, float* recons_attr) const {
    FAISS_THROW_IF_NOT(ni == 0 || (i0 >= 0 && i0 + ni <= ntotal));
    FAISS_THROW_IF_NOT(is_include_one_attribute);
    sa_one_attribute_decode(ni, attributes.data() + i0 * attr_size, recons_attr);
}

void IndexFlatCodes::reconstruct_n_two_attribute(idx_t i0, idx_t ni, float* recons_attr_first, float* recons_attr_second) const {
    FAISS_THROW_IF_NOT(ni == 0 || (i0 >= 0 && i0 + ni <= ntotal));
    FAISS_THROW_IF_NOT(is_include_two_attribute);
    sa_two_attribute_decode(ni, attributes_first.data() + i0 * attr_size, attributes_second.data() + i0 * attr_size,
                            recons_attr_first, recons_attr_second);
}

void IndexFlatCodes::reconstruct(idx_t key, float* recons) const {
    reconstruct_n(key, 1, recons);
}

void IndexFlatCodes::reconstruct_one_attribute(idx_t key, float* recons_attr) const {
    reconstruct_n_one_attribute(key, 1, recons_attr);
}

void IndexFlatCodes::reconstruct_two_attribute(idx_t key, float* recons_attr_first, float* recons_attr_second) const {
    reconstruct_n_two_attribute(key, 1, recons_attr_first, recons_attr_second);
}

FlatCodesDistanceComputer* IndexFlatCodes::get_FlatCodesDistanceComputer()
        const {
    FAISS_THROW_MSG("not implemented");
}

void IndexFlatCodes::check_compatible_for_merge(const Index& otherIndex) const {
    // minimal sanity checks
    const IndexFlatCodes* other = dynamic_cast<const IndexFlatCodes*>(&otherIndex);
    FAISS_THROW_IF_NOT(other);
    FAISS_THROW_IF_NOT(other->d == d);
    FAISS_THROW_IF_NOT(other->code_size == code_size);
    FAISS_THROW_IF_NOT(other->attr_size == attr_size);
    FAISS_THROW_IF_NOT_MSG(
            typeid(*this) == typeid(*other),
            "can only merge indexes of the same type");
    
    FAISS_THROW_IF_NOT_MSG(
            is_include_one_attribute == other->is_include_one_attribute,
            "both indexes must have same is_include_one_attribute status");

    FAISS_THROW_IF_NOT_MSG(
            is_include_two_attribute == other->is_include_two_attribute,
            "both indexes must have same is_include_two_attribute status");
}

void IndexFlatCodes::merge_from(Index& otherIndex, idx_t add_id) {
    FAISS_THROW_IF_NOT_MSG(add_id == 0, "cannot set ids in FlatCodes index");
    check_compatible_for_merge(otherIndex);
    IndexFlatCodes* other = static_cast<IndexFlatCodes*>(&otherIndex);
    codes.resize((ntotal + other->ntotal) * code_size);
    memcpy(codes.data() + (ntotal * code_size),
           other->codes.data(),
           other->ntotal * code_size);

    if(get_is_include_one_attribute()) {
        attributes.resize((ntotal + other->ntotal) * attr_size);
        memcpy(attributes.data() + (ntotal * attr_size),
               other->attributes.data(),
               other->ntotal * attr_size);
    }

    if(get_is_include_two_attribute()) {
        attributes_first.resize((ntotal + other->ntotal) * attr_size);
        attributes_second.resize((ntotal + other->ntotal) * attr_size);
        memcpy(attributes_first.data() + (ntotal * attr_size),
               other->attributes_first.data(),
               other->ntotal * attr_size);
        memcpy(attributes_second.data() + (ntotal * attr_size),
               other->attributes_second.data(),
               other->ntotal * attr_size);
    }

    ntotal += other->ntotal;
    other->reset();
}

CodePacker* IndexFlatCodes::get_CodePacker() const {
    FAISS_THROW_IF_NOT_MSG(is_include_one_attribute == false, "get_CodePacker for IndexFlatCodes not support for one attribute now");
    FAISS_THROW_IF_NOT_MSG(is_include_two_attribute == false, "get_CodePacker for IndexFlatCodes not support for two attribute now");
    return new CodePackerFlat(code_size);
}

void IndexFlatCodes::permute_entries(const idx_t* perm) {
    std::vector<uint8_t> new_codes(codes.size());
    std::vector<uint8_t> new_attributes(attributes.size());
    std::vector<uint8_t> new_attributes_first(attributes_first.size());
    std::vector<uint8_t> new_attributes_second(attributes_second.size());

    for (idx_t i = 0; i < ntotal; i++) {
        memcpy(new_codes.data() + i * code_size,
               codes.data() + perm[i] * code_size,
               code_size);
    }
    std::swap(codes, new_codes);


    if (get_is_include_one_attribute()) {
        for (idx_t i = 0; i < ntotal; i++) {
            memcpy(new_attributes.data() + i * attr_size,
                   attributes.data() + perm[i] * attr_size,
                   attr_size);
        }
        std::swap(attributes, new_attributes);
    }

    if (get_is_include_two_attribute()) {
        for (idx_t i = 0; i < ntotal; i++) {
            memcpy(new_attributes_first.data() + i * attr_size,
                   attributes_first.data() + perm[i] * attr_size,
                   attr_size);
            
            memcpy(new_attributes_second.data() + i * attr_size,
                   attributes_second.data() + perm[i] * attr_size,
                   attr_size);
        }
        std::swap(attributes_first, new_attributes_first);
        std::swap(attributes_second, new_attributes_second);
    }
}
}