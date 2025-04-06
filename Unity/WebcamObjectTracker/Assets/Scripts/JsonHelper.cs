using UnityEngine;
using System;

public static class JsonHelper
{
    [Serializable]
    private class Wrapper<T>
    {
        public T[] array;
    }

    public static T[] FromJson<T>(string json)
    {
        if (json.StartsWith("[") && json.EndsWith("]"))
        {
            // 배열을 객체로 래핑
            json = "{\"array\":" + json + "}";
            Wrapper<T> wrapper = JsonUtility.FromJson<Wrapper<T>>(json);
            return wrapper.array;
        }
        else
        {
            // 단일 객체인 경우
            T[] array = new T[1];
            array[0] = JsonUtility.FromJson<T>(json);
            return array;
        }
    }
}