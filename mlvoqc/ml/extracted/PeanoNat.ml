open Datatypes

module Nat =
 struct
  (** val add : int -> int -> int **)

  let rec add n m =
    (fun fO fS n -> if n=0 then fO () else fS (n-1))
      (fun _ -> m)
      (fun p -> Pervasives.succ (add p m))
      n

  (** val mul : int -> int -> int **)

  let rec mul n m =
    (fun fO fS n -> if n=0 then fO () else fS (n-1))
      (fun _ -> 0)
      (fun p -> add m (mul p m))
      n

  (** val sub : int -> int -> int **)

  let rec sub n m =
    (fun fO fS n -> if n=0 then fO () else fS (n-1))
      (fun _ -> n)
      (fun k ->
      (fun fO fS n -> if n=0 then fO () else fS (n-1))
        (fun _ -> n)
        (fun l -> sub k l)
        m)
      n

  (** val ltb : int -> int -> bool **)

  let ltb n m =
    (<=) (Pervasives.succ n) m

  (** val compare : int -> int -> comparison **)

  let rec compare = fun n m -> if n=m then Eq else if n<m then Lt else Gt

  (** val pow : int -> int -> int **)

  let rec pow n m =
    (fun fO fS n -> if n=0 then fO () else fS (n-1))
      (fun _ -> Pervasives.succ 0)
      (fun m0 -> mul n (pow n m0))
      m

  (** val divmod : int -> int -> int -> int -> int * int **)

  let rec divmod x y q u =
    (fun fO fS n -> if n=0 then fO () else fS (n-1))
      (fun _ -> (q, u))
      (fun x' ->
      (fun fO fS n -> if n=0 then fO () else fS (n-1))
        (fun _ -> divmod x' y (Pervasives.succ q) y)
        (fun u' -> divmod x' y q u')
        u)
      x

  (** val div : int -> int -> int **)

  let div x y =
    (fun fO fS n -> if n=0 then fO () else fS (n-1))
      (fun _ -> y)
      (fun y' -> fst (divmod x y' 0 y'))
      y

  (** val modulo : int -> int -> int **)

  let modulo x y =
    (fun fO fS n -> if n=0 then fO () else fS (n-1))
      (fun _ -> x)
      (fun y' -> sub y' (snd (divmod x y' 0 y')))
      y
 end
