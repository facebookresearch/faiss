#pragma once

#include <cstddef>
#include <cstdint>
#include <memory>
#include <vector>

#include <faiss/impl/FaissAssert.h>

namespace faiss {

// An interface for an owner of a MaybeOwnedVector.
struct MaybeOwnedVectorOwner {
    virtual ~MaybeOwnedVectorOwner() = default;
};

// a container that either works as std::vector<T> that owns its own memory,
//    or as a view of a memory buffer, with a known size
template <typename T>
struct MaybeOwnedVector {
    using value_type = T;
    using self_type = MaybeOwnedVector<T>;
    using vec_iterator = typename std::vector<T>::const_iterator;

    bool is_owned = true;

    // this one is used if is_owned == true
    std::vector<T> owned_data;

    // these three are used if is_owned == false
    T* view_data = nullptr;
    // the number of T elements
    size_t view_size = 0;
    // who owns the data.
    // This field can be nullptr, and it is present ONLY in order
    //   to avoid possible tricky memory / resource leaks.
    std::shared_ptr<MaybeOwnedVectorOwner> owner;

    // points either to view_data, or to owned.data()
    T* c_ptr = nullptr;
    // uses either view_size, or owned.size();
    size_t c_size = 0;

    MaybeOwnedVector() = default;
    MaybeOwnedVector(const size_t initial_size) {
        is_owned = true;

        owned_data.resize(initial_size);
        c_ptr = owned_data.data();
        c_size = owned_data.size();
    }

    MaybeOwnedVector(const std::vector<T>& vec)
            : MaybeOwnedVector<T>(vec.size()) {
        if (!vec.empty()) {
            memcpy(owned_data.data(), vec.data(), vec.size() * sizeof(T));
        }
    }

    MaybeOwnedVector(const MaybeOwnedVector& other) {
        is_owned = other.is_owned;
        owned_data = other.owned_data;

        view_data = other.view_data;
        view_size = other.view_size;
        owner = other.owner;

        if (is_owned) {
            c_ptr = owned_data.data();
            c_size = owned_data.size();
        } else {
            c_ptr = view_data;
            c_size = view_size;
        }
    }

    MaybeOwnedVector(MaybeOwnedVector&& other) {
        is_owned = other.is_owned;
        owned_data = std::move(other.owned_data);

        view_data = other.view_data;
        view_size = other.view_size;
        owner = std::move(other.owner);
        other.owner = nullptr;

        if (is_owned) {
            c_ptr = owned_data.data();
            c_size = owned_data.size();
        } else {
            c_ptr = view_data;
            c_size = view_size;
        }
    }

    MaybeOwnedVector& operator=(const MaybeOwnedVector& other) {
        if (this == &other) {
            return *this;
        }

        // create a copy
        MaybeOwnedVector cloned(other);
        // swap
        swap(*this, cloned);

        return *this;
    }

    MaybeOwnedVector& operator=(MaybeOwnedVector&& other) {
        if (this == &other) {
            return *this;
        }

        // moved
        MaybeOwnedVector moved(std::move(other));
        // swap
        swap(*this, moved);

        return *this;
    }

    MaybeOwnedVector(std::vector<T>&& other) {
        is_owned = true;

        owned_data = std::move(other);
        c_ptr = owned_data.data();
        c_size = owned_data.size();
    }

    static MaybeOwnedVector create_view(
            void* address,
            const size_t n_elements,
            const std::shared_ptr<MaybeOwnedVectorOwner>& owner) {
        MaybeOwnedVector vec;
        vec.is_owned = false;
        vec.view_data = reinterpret_cast<T*>(address);
        vec.view_size = n_elements;
        vec.owner = owner;

        vec.c_ptr = vec.view_data;
        vec.c_size = vec.view_size;

        return vec;
    }

    const T* data() const {
        return c_ptr;
    }

    T* data() {
        return c_ptr;
    }

    size_t size() const {
        return c_size;
    }

    T& operator[](const size_t idx) {
        return c_ptr[idx];
    }

    const T& operator[](const size_t idx) const {
        return c_ptr[idx];
    }

    vec_iterator begin() const {
        FAISS_ASSERT_MSG(
                is_owned,
                "This operation cannot be performed on a viewed vector");

        return owned_data.begin();
    }

    vec_iterator end() const {
        FAISS_ASSERT_MSG(
                is_owned,
                "This operation cannot be performed on a viewed vector");

        return owned_data.end();
    }

    vec_iterator erase(vec_iterator begin, vec_iterator end) {
        FAISS_ASSERT_MSG(
                is_owned,
                "This operation cannot be performed on a viewed vector");

        return owned_data.erase(begin, end);
    }

    void clear() {
        FAISS_ASSERT_MSG(
                is_owned,
                "This operation cannot be performed on a viewed vector");

        owned_data.clear();
        c_ptr = owned_data.data();
        c_size = owned_data.size();
    }

    void resize(const size_t new_size) {
        FAISS_ASSERT_MSG(
                is_owned,
                "This operation cannot be performed on a viewed vector");

        owned_data.resize(new_size);
        c_ptr = owned_data.data();
        c_size = owned_data.size();
    }

    void resize(const size_t new_size, const value_type v) {
        FAISS_ASSERT_MSG(
                is_owned,
                "This operation cannot be performed on a viewed vector");

        owned_data.resize(new_size, v);
        c_ptr = owned_data.data();
        c_size = owned_data.size();
    }

    friend void swap(self_type& a, self_type& b) {
        std::swap(a.is_owned, b.is_owned);
        std::swap(a.owned_data, b.owned_data);
        std::swap(a.view_data, b.view_data);
        std::swap(a.view_size, b.view_size);
        std::swap(a.owner, b.owner);
        std::swap(a.c_ptr, b.c_ptr);
        std::swap(a.c_size, b.c_size);
    }
};

template <typename T>
struct is_maybe_owned_vector : std::false_type {};

template <typename T>
struct is_maybe_owned_vector<MaybeOwnedVector<T>> : std::true_type {};

template <typename T>
inline constexpr bool is_maybe_owned_vector_v = is_maybe_owned_vector<T>::value;

} // namespace faiss